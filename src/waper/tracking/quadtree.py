from operator import add
from functools import reduce
import networkx as nx
import sys

# function to split input raster image into 4 equal images 
# returns the 4 split images 

def split4(image):
    half_split = np.array_split(image, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)

# function to compute mean pixel value of an image
# returns mean value for each image

def calculate_mean(img):
    return np.mean(img, axis=(0, 1))

# function to return all the features (in terms of pixel value assigned to each feature) present in an image
def feature(image,array):
    f=[]
    r,c = image.shape
    for i in range(len(array)):
        for j in range(r):
            for k in range(c):
                if(array[i] == image[j][k]):
                    if(array[i] not in f):
                        f.append(array[i])
                elif(image[j][k]==0):
                    if(0 not in f):
                        f.append(0)
    f = np.array(f)                
    return f
    

# function to create a quadtree corresponding to a raster image. Initial node i=0 is created beforehand and passed on as an argument
# i represents the position of a node in the raster image. Between 2 quadtrees, nodes with the same i value represents the same position within their respective raster images
# level represents the height at which a particular node is located in a quadtree. It serves as a pointer to the dimensions of a node (how many pixels it contains)
# the function returns the constructed quadtree Q; nodes are given by their i values and each node consists of the mean of the pixel values, features it contains and the height/level at which the node is located

def insert_node(Q,i,image,value,level):
    r,c = np.array(image).shape
    if(r>1 and c>1):
        level = level+1
        split_img = split4(image)
        split_img = np.array(split_img)

        m_1 = calculate_mean(split_img[0])
        f_1 = feature(split_img[0],a[value])
        Q.add_node((4*i)+1,mean = m_1, f = f_1,level = level)
        Q.add_edge(i,(4*i)+1)

        m_2 = calculate_mean(split_img[1]) 
        f_2 = feature(split_img[1],a[value])
        Q.add_node((4*i)+2,mean = m_1, f = f_2,level = level)
        Q.add_edge(i,(4*i)+2)

        m_3 = calculate_mean(split_img[2])
        f_3 = feature(split_img[2],a[value])
        Q.add_node((4*i)+3,mean = m_1, f = f_3,level = level)
        Q.add_edge(i,(4*i)+3)

        m_4 = calculate_mean(split_img[3])
        f_4 = feature(split_img[3],a[value])
        Q.add_node((4*i)+4,mean = m_1, f = f_4,level = level)
        Q.add_edge(i,(4*i)+4)

        if(len(f_1)>1):
            Q = insert_node(Q,(4*i)+1,split_img[0],value,level)

        if(len(f_2)>1):
            Q = insert_node(Q,(4*i)+2,split_img[1],value,level)

        if(len(f_3)>1):
            Q = insert_node(Q,(4*i)+3,split_img[2],value,level)

        if(len(f_4)>1):
            Q = insert_node(Q,(4*i)+4,split_img[3],value,level)


    return Q

