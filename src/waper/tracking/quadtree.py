from operator import add
from functools import reduce
import networkx as nx
import math
import numpy as np
from collections import defaultdict

from .rwp_polygon import WAPER_NUM_PIXELS, WAPER_IMAGE_SIZE

# function to split input raster image into 4 equal images
# returns the 4 split images


def split4(raster):
    # half_split = np.array_split(raster, 2)
    # res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    # return np.array(reduce(add, res))
    r, c = raster.shape

    half_r = int(r / 2)
    half_c = int(c / 2)

    return (
        raster[:half_r, :half_c],
        raster[:half_r, half_c:],
        raster[half_r:, :half_c],
        raster[half_r:, half_c:],
    )


# function to compute mean pixel value of an image
# returns mean value for each image


def calculate_mean(raster):
    return np.mean(raster)


# function to return all the features (in terms of pixel value assigned to each feature) present in an image
def get_features(raster):
    features = set(raster.ravel())
    # features.add(0)
    return tuple(features)


# function to create a quadtree corresponding to a raster image. Initial node i=0 is created beforehand and passed on as an argument
# i represents the position of a node in the raster image. Between 2 quadtrees, nodes with the same i value represents the same position within their respective raster images
# level represents the height at which a particular node is located in a quadtree. It serves as a pointer to the dimensions of a node (how many pixels it contains)
# the function returns the constructed quadtree Q; nodes are given by their i values and each node consists of the mean of the pixel values, features it contains and the height/level at which the node is located


def create_quadtree(raster):
    quadtree = nx.DiGraph()

    quadtree.add_node(
        0, mean=np.mean(raster), features=get_features(raster), level=0, start_pixel=(0, 0)
    )

    return insert_node(quadtree, 0, raster, 0)


def insert_node(Q, parent_node_id, raster, level):

    r, c = np.array(raster).shape
    if r > 1 and c > 1:
        level = level + 1
        split_raster = split4(raster)

        parent_start_x, parent_start_y = Q.nodes[parent_node_id]["start_pixel"]

        m_1 = calculate_mean(split_raster[0])
        f_1 = get_features(split_raster[0])
        Q.add_node(
            (4 * parent_node_id) + 1,
            mean=m_1,
            features=f_1,
            level=level,
            start_pixel=(parent_start_x, parent_start_y),
        )
        Q.add_edge(parent_node_id, (4 * parent_node_id) + 1)

        m_2 = calculate_mean(split_raster[1])
        f_2 = get_features(split_raster[1])
        Q.add_node(
            (4 * parent_node_id) + 2,
            mean=m_2,
            features=f_2,
            level=level,
            start_pixel=(parent_start_x, parent_start_y + int(r / 2)),
        )
        Q.add_edge(parent_node_id, (4 * parent_node_id) + 2)

        m_3 = calculate_mean(split_raster[2])
        f_3 = get_features(split_raster[2])
        Q.add_node(
            (4 * parent_node_id) + 3,
            mean=m_3,
            features=f_3,
            level=level,
            start_pixel=(parent_start_x + int(r / 2), parent_start_y),
        )
        Q.add_edge(parent_node_id, (4 * parent_node_id) + 3)

        m_4 = calculate_mean(split_raster[3])
        f_4 = get_features(split_raster[3])
        Q.add_node(
            (4 * parent_node_id) + 4,
            mean=m_4,
            features=f_4,
            level=level,
            start_pixel=(parent_start_x + int(r / 2), parent_start_y + int(r / 2)),
        )
        Q.add_edge(parent_node_id, (4 * parent_node_id) + 4)

        if len(f_1) > 1:
            Q = insert_node(Q, (4 * parent_node_id) + 1, split_raster[0], level)

        if len(f_2) > 1:
            Q = insert_node(Q, (4 * parent_node_id) + 2, split_raster[1], level)

        if len(f_3) > 1:
            Q = insert_node(Q, (4 * parent_node_id) + 3, split_raster[2], level)

        if len(f_4) > 1:
            Q = insert_node(Q, (4 * parent_node_id) + 4, split_raster[3], level)

    return Q


# function to compute the number of pixels corresponding to each feature in a particular quadtree
# returns a dictionary "pixel_dict" whose - key:feature values, values:number of pixels


def compute_pixels(quadtree):

    pixel_dict = defaultdict(list)
    leaf_nodes = [
        node
        for node in quadtree.nodes()
        if quadtree.in_degree(node) != 0 and quadtree.out_degree(node) == 0
    ]
    for i in range(len(leaf_nodes)):
        f = quadtree.nodes[leaf_nodes[i]]["features"]
        f = tuple(f)
        if 0 in f:
            continue
        if f in pixel_dict:
            pixel_dict[f] += WAPER_NUM_PIXELS / (4 ** (quadtree.nodes[leaf_nodes[i]]["level"]))
        else:
            pixel_dict[f] = WAPER_NUM_PIXELS / (4 ** (quadtree.nodes[leaf_nodes[i]]["level"]))
    return pixel_dict


def contains_no_features(node):
    if len(node["features"]) == 1:
        if node["features"][0] == 0:
            return True

    return False


def contains_more_than_one_feature(node):
    return len(node["features"]) > 1


def contains_one_feature(node):
    if len(node["features"]) == 1:
        if node["features"][0] != 0:
            return True

    return False


# function to construct a certain branch of the merge quadtree
# Used when in a particular location, the feature in one quadtree is bigger in size compared to the feature in the second quadtree
# G represents quadtree who has the smaller feature; i is the leaf node(feature node) in the other quadtree
# Returns merged quadtree Q with the branch rooted at i same as that of the input quadtree G

def construct(merged_quadtree, test_quadtree, node_number, larger_feature):

    for j in range(1, 5):
        if contains_no_features(test_quadtree.nodes[(4 * node_number) + j]):
            merged_quadtree.add_node((4 * node_number) + j, features=[0], level=test_quadtree.nodes[(4 * node_number) + j]["level"],
                       start_pixel=test_quadtree.nodes[(4 * node_number) + j]["start_pixel"])
        else:
            merged_quadtree.add_node(
                (4 * node_number) + j,
                features=np.sort(np.concatenate([test_quadtree.nodes[(4 * node_number) + j]["features"], larger_feature])),
                level=test_quadtree.nodes[(4 * node_number) + j]["level"], start_pixel=test_quadtree.nodes[(4 * node_number) + j]["start_pixel"]
            )
        merged_quadtree.add_edge(node_number, (4 * node_number) + j)

    for j in range(1, 5):
        if len(test_quadtree.nodes[(4 * node_number) + j]["features"]) > 1:
            merged_quadtree = construct(merged_quadtree, test_quadtree, (4 * node_number) + j, larger_feature)
    return merged_quadtree


# function to merge two quadtrees G and H; Q represents the merged quadtree
# returns the merged quadtree Q
def merge(prev_time_quadtree, curr_time_quadtree):
    
    merged_quadtree = nx.DiGraph()
    common_nodes = set(curr_time_quadtree).intersection(prev_time_quadtree)
    for node_number in common_nodes:
        if node_number not in list(merged_quadtree):
            if contains_no_features(prev_time_quadtree.nodes[node_number]) or contains_no_features(
                curr_time_quadtree.nodes[node_number]
            ):
                merged_quadtree.add_node(
                    node_number, features=[0], level=prev_time_quadtree.nodes[node_number]["level"],
                    start_pixel=prev_time_quadtree.nodes[node_number]["start_pixel"]
                )
                if math.ceil((node_number / 4) - 1) >= 0:
                    merged_quadtree.add_edge(math.ceil((node_number / 4) - 1), node_number)

            elif contains_more_than_one_feature(
                prev_time_quadtree.nodes[node_number]
            ) and contains_one_feature(curr_time_quadtree.nodes[node_number]):
                merged_quadtree = construct(merged_quadtree, prev_time_quadtree, node_number, curr_time_quadtree.nodes[node_number]["features"])

            elif contains_more_than_one_feature(
                curr_time_quadtree.nodes[node_number]
            ) and contains_one_feature(prev_time_quadtree.nodes[node_number]):
                merged_quadtree = construct(merged_quadtree, curr_time_quadtree, node_number, prev_time_quadtree.nodes[node_number]["features"])

            else:
                features = np.concatenate(
                    [
                        prev_time_quadtree.nodes[node_number]["features"],
                        curr_time_quadtree.nodes[node_number]["features"],
                    ]
                )
                merged_quadtree.add_node(
                    node_number,
                    features=np.sort(features),
                    level=prev_time_quadtree.nodes[node_number]["level"],
                    start_pixel=prev_time_quadtree.nodes[node_number]["start_pixel"]
                )
                if math.ceil((node_number / 4) - 1) >= 0:
                    merged_quadtree.add_edge(math.ceil((node_number / 4) - 1), node_number)
                    
    return merged_quadtree

def reconstruct_image(quadtree):

    image = np.zeros((WAPER_IMAGE_SIZE, WAPER_IMAGE_SIZE))

    leaf_nodes = [
        node
        for node in quadtree.nodes()
        if quadtree.in_degree(node) != 0 and quadtree.out_degree(node) == 0
    ]

    for node in leaf_nodes:
        feature = quadtree.nodes[node]["features"][0]
        level = quadtree.nodes[node]["level"]
        x_pixel, y_pixel = quadtree.nodes[node]["start_pixel"]
        image[x_pixel:x_pixel + int(WAPER_IMAGE_SIZE/(2**level)), y_pixel:y_pixel + int(WAPER_IMAGE_SIZE/(2**level))] = feature
        
    return image
