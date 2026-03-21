Identification and Tracking of Rossby
Wave Packets
A Thesis

submitted to
Indian Institute of Science Education and Research Pune
in partial fulfillment of the requirements for the
BS-MS Dual Degree Programme
by

Malavika Biju

Indian Institute of Science Education and Research Pune
Dr. Homi Bhabha Road,
Pashan, Pune 411008, INDIA.

May, 2022

Supervisor: Dr. Joy Merwin Monteiro
© Malavika Biju 2022
All rights reserved

Certificate
This is to certify that this dissertation entitled ”Identification and Tracking of Rossby
Wave Packets” towards the partial fulfilment of the BS-MS dual degree programme at the
Indian Institute of Science Education and Research, Pune represents study/work carried
out by Malavika Biju at Indian Institute of Science Education and Research under the
supervision of Dr. Joy Merwin Monteiro, Assistant Professor, Department of ECS , during
the academic year 2021-2022.

Dr. Joy Merwin Monteiro

Committee:
Dr. Joy Merwin Monteiro
Dr. Suhas Ettammal

This thesis is dedicated to my parents.

Declaration
I hereby declare that the matter embodied in the report entitled ”Identification and
Tracking of Rossby Wave Packets”, are the results of the work carried out by me at the
Department of ECS, Indian Institute of Science Education and Research, Pune, under the
supervision of Dr. Joy Merwin Monteiro and the same has not been submitted elsewhere
for any other degree.

Malavika Biju

vi

Acknowledgments
First and foremost, I thank my supervisor, Dr. Joy Monteiro. His guidance and invaluable
support have helped me conduct original work in the field of climate sciences, to which I was
completely new. Special thanks to him for forgiving all my amateur mistakes. I also thank
my co-supervisor, Dr. Vijay Natarajan, whose suggestions and guidance have immensely
helped me during the second phase of my work. The advice given by my TAC member Dr.
Suhas Ettammal has helped me improve my domain knowledge.
My friends and family have been a constant support throughout my BS-MS journey at
IISER, Pune. I thank my friends, Yashi Jain and Shweta Singh, for always being up for
discussions on mathematics, broader goals, and opportunities in academia and otherwise. I
thank my friend Suhail Odungat for being a pillar of mental support throughout the pandemic and other difficult times.
Finally, I thank INSPIRE (Innovation in Science Pursuit for Inspired Research) for funding
my study at IISER, Pune

vii

Abstract
Rossby Wave Packets (RWPs) are regions of high amplitude meridional winds in the upper troposphere that acts as precursors to extreme weather phenomena. Due to this, the
identification and tracking of RWPs is an important field of research. We have implemented
modifications to an identification algorithm that uses the underlying topology of the meridional wind field and techniques of computational geometry to identify RWPs from gridded
data. These modifications have helped improve efficiency and reduce computational complexity. We have also implemented a feature-based tracking approach to track the evolution
of RWPs across time. We recognize regions enclosing the identified RWPs as features. These
features are then stored in the form of quadtrees for each time step which optimises storage and facilitates easy computation of unions and sizes of features. We then compute the
overlap of features in consecutive time steps using their respective quadtree representations
and formulate a weight that determines the extent of each overlap. Finally, we construct
a tracking graph consisting of nodes representing features in each timestep and weighted
edges representing the extent of overlap between these features. This representation allows
for easy analysis of RWP tracks and contains information about splits and merges of RWP
tracks.

viii

Contents

Abstract

viii

1 Introduction

1

1.1

Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

1

1.2

Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

2

1.3

Topological Methods in Climate Visualisation . . . . . . . . . . . . . . . . .

4

2 Identification and Visualisation of RWPs

7

2.1

The Identification problem . . . . . . . . . . . . . . . . . . . . . . . . . . . .

7

2.2

Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

8

2.3

Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

8

2.4

Modifications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

13

3 Tracking of RWPs

17

3.1

The Tracking problem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

17

3.2

Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

18

4 Results and Conclusions
4.1

Feature Identification Calculations

25
. . . . . . . . . . . . . . . . . . . . . . .
ix

25

4.2

Tracking Graph Calculations . . . . . . . . . . . . . . . . . . . . . . . . . . .

25

4.3

Discussion and Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . .

28

x

List of Figures
1.1

The Hovmöller diagram shows the 250-hPa meridional wind (the colour bar
is in m/s) during the August 2002 time period, using data averaged between
latitudes 40° and 60°N. The dates are July 30 – August 15, 2002. The map
on top of the illustration aids spatial navigation.[WRCM18] . . . . . . . . .

3

2.1

Local Maxima of the 300hPa meridional wind field on 21 Jan 2007 (0600 UTC)

9

2.2

Local Minima of the 300hPa meridional wind field on 21 Jan 2007 (0600 UTC)

9

2.3

Clustering applied to 300hPa meridional wind field on 21 Jan 2007 (0600
UTC). Black points represent filtered critical points. Red shapes represent
minima clusters while green shapes represent maxima clusters. . . . . . . . .

10

Association graph obtained for 300hPa meridional wind field on 21 Jan 2007
(0600 UTC).The blue points represent minima clusters and the red points represent maxima clusters; The solid purple line represents the zero-isocontour;
The green line represents associations between maxima and minima clusters.

11

Scalar Pruned association graph obtained for 300hPa meridional wind field
on 21 Jan 2007 (0600 UTC) . . . . . . . . . . . . . . . . . . . . . . . . . . .

12

Edge Pruned association graph obtained for 300hPa meridional wind field on
21 Jan 2007 (0600 UTC) . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

12

Final RWPs obtained for 300hPa meridional wind field on 21 Jan 2007 (0600
UTC) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

13

Clustering using different diagonal weights in the affinity propagation clustering algorithm. Clusters are computed for meridional wind field at 300hPa
(0600 UTC 21 January 2007) . . . . . . . . . . . . . . . . . . . . . . . . . .

15

2.4

2.5

2.6

2.7

2.8

xi

3.1

(a) Regions of RWP activity (features) identified for 21 January 2007 (0600
UTC) (b) Rasterised image with features identified for 21 January 2007 (0600
UTC) burned in . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

19

3.2

Quadtree representation of an image of dimension 8x8[Mu8] . . . . . . . . .

21

3.3

Evolutionary events that occur in time-varying datasets. Polygons in solid
black line represents RWP polygons identified in the current timestep and
the polygons in dotted black line represents RWP polygons identified in the
previous timestep . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

23

Regions identified for individual RWPs using different factors. The panel
shows meridional wind field at 300hPa (0600 UTC 20 January 2007). The
bold black lines denote the representative RWPs in that timestep and the
polygons in the thin black lines denote the identified feature . . . . . . . . .

26

Regions identified for individual RWPs using different factors. The panel
shows meridional wind field at 300hPa (0600 UTC 12 January - 18 January
2007). Black points represent centroids of features identified in the current
timestep. Red points represent centroids of features identified in the previous
timesteps. The solid black lines connecting red and black points denote the
movement of the RWP from the previous timestep to the current. The dotted
black lines connecting red and red points denote all the previous movements
of the RWP. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

29

Regions identified for individual RWPs using different factors. The panel
shows meridional wind field at 300hPa (0600 UTC 20 January - 26 January
2007) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

30

4.1

4.2

4.3

xii

Chapter 1
Introduction

1.1

Background

A Rossby Wave Packet (RWP) is a localized region of high amplitude meridional flow in
the upper troposphere [SCC14a]. Its central characteristic is the propagation in the zonal
direction. It also has a group velocity greater than the phase velocity, which leads to the
transfer of wave energy from one individual trough or ridge to its downstream neighbor, a
process called “downstream development”[WRCM18].
There are many different meteorological processes that can cause an RWP. Some of these
include diabatic heating in the mid–lower troposphere due to pre-existing synoptic disturbances such as extratropical cyclones and recycling from earlier waves in the jet-stream
waveguide.The strength and localization of the large-scale background potential vorticity
(PV) gradient, which functions as a waveguide, is largely responsible for RWP propagation
and extension.Such coherent, long RWPs are more common in the Southern Hemisphere. In
the Northern Hemisphere, RWPs are often shorter and more difficult to trace due to the
existence of broad, baroclinically unfavourable continental expanses.[GV]
There has been an increasing interest in the RWPs along the midlatitude waveguide. The
major reason for this is that these RWPs often act as precursors to extreme weather events
in the midlatitudes. Extreme weather events are events based on local recorded weather
1

history and lie in the tail of the respective local distribution. Hence a better understanding
of RWP dynamics would help in improving the prediction of these extremes. RWPs have
also been linked to large-scale regime changes and downstream propagation of model errors
and uncertainties[Hak05][ABKC13].

1.2

Related Work

The importance of RWPs has motivated many studies to provide a precise algorithmic description to track and identify RWPs. In this section, we discuss the diagnostic frameworks
that have been developed earlier for the study of RWP dynamics.

1.2.1

Hovmöller Diagrams

The Hovmöller diagram, designed in 1948 by Ernst Hovmöller, is a diagram in which the latitudinal average of a variable from the upper troposphere (such as geopotential, meridional
wind, meridional wind squared, etc.) is plotted in a longitude-time map [WRCM18]. This
method condenses the spatiotemporal information and makes use of the excellent pattern
recognition skills of the human brain to track individual RWPs across time. The main limitation of this method is that since it uses a latitudinal average, we cannot obtain information
about the latitudinal propagation of the RWPs. This would limit the application of these
diagrams to long time series as RWPs may shift latitudinally with time (Eg: as a function of
seasons). There have been many modifications to overcome the shortcomings of the classical
Hovmöller diagrams. For example, [GDJ+ 11] proposes the use of a latitudinal weighting
function that favors those latitudes where the longitudinal variance of the variable is large
instead of averaging over a fixed latitude band. However, these methods cannot detect the
two-dimensional horizontal structure of the wave packets and can lead to erroneous results
when multiple RWPs exist in close proximity at different latitudes.
2

Figure 1.1: The Hovmöller diagram shows the 250-hPa meridional wind (the colour bar is
in m/s) during the August 2002 time period, using data averaged between latitudes 40° and
60°N. The dates are July 30 – August 15, 2002. The map on top of the illustration aids
spatial navigation.[WRCM18]

1.2.2

Envelope Reconstruction Methods

We require a more objective and automated approach than the previous for identification
and tracking of RWPs from large datasets. In this direction envelope reconstruction methods
such as complex demodulation, Hilbert transform based methods have been developed.
• Complex demodulation: This technique assumes a spatially dependent scalar atmospheric quantity, v(x) can be written in the form of v(x) = Re[A(x)exp(ikx)] where
k is a supposed carrier wavenumber. Then we demodulate by multiplying v(x) with
exp(−ikx) and finally extract the absolute value of the wave packet envelope, |A(x)|
using low-pass filtering to remove higher wavenumber components. Though we generally obtain reasonable results when carrier wavenumber is chosen from a plausible
range of values, this method would yield incorrect results when multiple wave packets
of different carrier wavenumbers coexist in the same latitude[ZSP+ 03].
• Hilbert transform based technique: [ZSP+ 03] suggested an alternative technique to
extract the envelope of wave packets that is not affected by the aforementioned problem.
It involves Hilbert transform along a latitudinal circle. Unlike complex demodulation,
this method does not require a prespecified carrier wavenumber.
3

However, both these envelope reconstruction methods implicitly assume that RWPs are
purely oriented in the zonal direction and are almost plane wave packets which are not always true. Furthermore, they also assume that the wave packets must be small amplitude
to allow linearization. However, observed RWPs are often large amplitude and often display
non linear behaviors such as wave-breaking and cutoff-formation, which contradicts the above
assumption. To overcome these shortcomings [GFW18] introduced Local finite-amplitude
wave activity (LWA), a diagnostic method to identify RWPs and quantify their amplitude.
The above mentioned algorithms provide a description of RWPs at the expense of losing
phase information. However, the phase information is often important for characterizing the
local synoptic situation at a location.
Combining the identification step with a suitable tracking algorithm would help us to effectively extract information about RWPs from large datasets.[SCC14b] describes featurebased tracking techniques to track RWPs. They use two objective approaches for tracking:
a point-based cost optimisation method and a hybrid method involving both point-based
and object-based methods. Point-based tracking method involves identifying the significant
relative maxima and minima points in the atmospheric data as objects to be tracked. On
the other hand, object-based tracking method considers the entire enclosed shape of a wave
packet as the object to be tracked and uses approaches such as statistical similarities, overlap
or object proximity between two time steps for tracking. The hybrid tracking approach proposed in [SCC14b] uses both the local extrema of RWP amplitudes and the overall structure
of the wave packet.

1.3

Topological Methods in Climate Visualisation

Topology-based methods have proved to be a versatile approach to analysing scientific data.
There are numerous topological models that can be used for the analysis of scalar fields.
These include critical points and persistent homology, merge trees, contour trees, Reeb
graphs, Morse Smale Complexes, and extremum graphs. We discuss a few important topological concepts below:

• Level Sets: Level sets are important topological descriptors of a dataset. The level set
4

of a function f : Ω → R is its preimage for some value ν:

f −1 (ν) = x ∈ Ω|f (x) = ν
This preimage would consist of multiple connected components which are called isocontours. We define the concept called Reeb Graph using the idea of isocontours. Similar
to level sets, we can also define sublevel and superlevel sets of a function. The sublevel

set of a function f : Ω → R for a value ν is given by x ∈ Ω|f (x) ≤ ν . Similarly, the

superlevel set of f for a value ν is given by x ∈ Ω|f (x) ≥ ν .
• Morse Theory: It is a mathematical theory that enables the study of the topology of a
domain using functions that are defined on that domain. To understand Morse Theory,
let us first define a Morse function. Let M be a smooth manifold and f : M → R a
smooth function. The points in M where the partial derivatives of f vanish are called
critical points and their images are called critical values. A critical point x ∈ M is
said to be non-degenerate if the matrix of second partial derivatives (Hessian matrix)
of f is non-singular. If all the critical values of f are non-degenerate and have distinct
function values, f is a Morse function. A main result of Morse theory is that the set
of Morse functions is an open dense subset of the space of smooth functions. In other
words, almost all smooth functions are Morse functions. Two fundamental theorems
of Morse theory is about the study of changes of the sublevel sets of a function with
respect to its critical points. Briefly, these theorems indicate if and when the topology
of the sublevel sets Mt vary as t changes.
However, most of the time we are not given a smooth manifold M. We are given a point
cloud sample of M. Therefore, in these cases, we use combinatorial structures (simplicial complexes) on the sample point cloud as an approximation of the underlying manifold M. Examples of simplicial complexes are triangulations, lattices, quadrangular
meshes etc. In this case, we define a piece-wise linear function on this complex and use
ideas from Morse Theory on this piece-wise linear approximation[YMS+ 21][RKG+ 11].
• Merge Graph: A merge graph (tree) is a tool to study the changes in connectivity in
the sublevel sets of a Morse function ff : M → R (where M is connected). We define
two pointsx, y ∈ M to be equivalent (w.r.t f) if x, y belong to the same connected
component of Mt for some t ∈ R. A connected component is represented by a node
in the tree and whenever a distinct connected component appears as t varies, a new
5

node is added to the tree. Therefore, this tree helps us to track critical points as well
as record the evolution of the connectivity of Mt as t changes. Further discussion on
computation of merge trees for piece-wise linear functions can be found in [HLH+ 16].
• Reeb Graph: Reeb graphs, on the other hand, are used to study the changes in connectivity in the level sets of a Morse function ff : M → R (where M is connected).
We define two pointsx, y ∈ M to be equivalent (w.r.t f) if f (x) = f (y) = t and x, y
belong to the same connected component of f −1 (t) for some t ∈ R. A Reeb graph is a
quotient space on M/ ∼, where the equivalence relation is as described above. If the
domain of f is simply connected, then the Reeb graph would have a tree structure and
is called a contour tree. Similar to merge trees, Reeb graphs helps us to record the
evolution of the connectivity of the level sets of f as t changes. Further discussion on
computation of Reeb graphs and contour trees can be found in [HLH+ 16].
Topological techniques have been widely used in climate sciences for feature tracking. In
most cases, feature-tracking involves the identification of topological features in each dataset
and matching them across time using a correspondence mechanism. [RKWH12] tracks critical points of 2D time dependent scalar fields using a combinatorial algorithm. This is done
by converting the scalar field at each time step into a 3D combinatorial vector field and using
a path search for the identification and tracking of critical points. This method is robust
and computationally fast and has been successfully used for vortex tracking by [KHNH12].
[VMN+ 19] uses a topological pipeline for the identification and tracking of cyclones. This
method first identifies cyclone centers within each time slice using local minima of the mean
sea level pressure. To produce the final tracks, candidate tracks are calculated from an optical flow field, which is then clustered in a moving time window.
[DNN13] describes a method for exploration and tracking of cloud systems using infrared
brightness temperatures. Merge trees are used for the computation and exploration of clouds.
The threshold for clouds of interest is determined using a persistence diagram. Cloud movement is then tracked using an optical flow field computation. A tracking graph is then
constructed using the optical flow between the computed clouds.

6

Chapter 2
Identification and Visualisation of
RWPs
In this chapter, we discuss the formulation of the identification problem along with the details
of the algorithm proposed in [PMN20] and the modifications made to the algorithm.

2.1

The Identification problem

As discussed earlier, Identification of RWPs using Fourier transform techniques results in the
loss of phase information which is often important for characterizing RWPs. Hence [PMN20]
proposes an algorithm for the identification of RWPs using the topology of the meridional
wind field to obtain a concise description of RWPs in the form of a graph without using
Fourier transforms. The nodes of this graph represent individual wave components in the
form of maxima and minima clusters, and the edges represent spatial adjacency between
the clusters. The maxima (minima) clusters consist of local maxima (minima) points of
the wind field. This approach does not make any assumption about the wavelike nature of
RWPs hence does not result in the loss of phase information. The graph representation also
enables fast computation and recomputation of RWPs for different parameter values.
7

2.2

Data

All datasets are from ECMWF interim data downloaded in the form of NetCDF files. We use
the 300 hPa meridional winds, v, with a spatial resolution of 0.75◦ in latitude and longitude
space and a temporal resolution of 6 h. Data is available at each grid point, and we assume
linear interpolation along axes for all computations. The algorithm does not assume a specific
resolution or projection of the input data. We use the python library NetworkX to store and
process graphs. The maxima and minima clusters, along with their associations, are stored
as a list of nodes and edges using the NetworkX library.

2.3

Methodology

In this section, we provide the details of each step in the identification algorithm.
• Identifying the critical points: Critical point analysis is an important technique in
topological data analysis. We also saw many examples in the previous chapter where
critical point analysis has been successfully used in climate sciences. [PMN20] use a
combinatorial approach for the computation of extrema instead of using gradients to
find the critical points. The local extrema are located at points whose grid value is
higher/lower than its neighbouring points. We are interested in capturing the clusterlike behavior of the extrema in the meridional wind field v. To avoid the detection of
insignificant critical points, we apply a filter of 5m/s to remove all local maxima and
minima below this value.
Figure 2.1 shows the initial local maxima identified in the northern hemisphere on
21 January 2007 before the 5m/s filter is applied. Similarly, figure 2.2 shows the initial
local minima identified in the northern hemisphere on 21 January 2007 before this filter
is applied.
• Computing maxima and minima clusters: The critical points located in the previous
step has to be divided into maxima and minima clusters. Each cluster represents a
potential component of an RWP. The maxima clusters are a set of proximal local maxima that lies on the same connected components of the superlevel set of 0. Similarly,
8

Figure 2.1: Local Maxima of the 300hPa meridional wind field on 21 Jan 2007 (0600 UTC)

Figure 2.2: Local Minima of the 300hPa meridional wind field on 21 Jan 2007 (0600 UTC)

minima clusters are a set of proximal local minima that lies on the same connected
components of the sublevel set of 0. For the computation of this partition, we use the
affinity propogation algorithm[FD07] along with a similarity measure that takes both
these requirements into account [FD07].
The affinity propagation algorithm works by recursively exchanging messages between
the data points until a good set of stable clusters and corresponding representative
member values emerge. The messages are based on the suitability of an input point
to become an exemplar (representative point) for another point in the dataset. The
algorithm takes as input the similarities between the data points, which denotes how
well one data point can represent the other, and “preferences”. Preferences are the
diagonal values of the similarity matrix, specified by the user to control the number of
clusters. If a point has a higher diagonal value, then it has a higher chance of being
picked as an exemplar. If at the beginning all points are equally likely to be exemplars,
the diagonal similarity should be set to the same value for every point in the dataset.
This value would influence the number of clusters. For example, taking the median
of all similarities between pairs of points would yield a reasonable number of clusters,
while taking the minimum of the similarities would result in a reduced number of clusters [FD07].

9

Figure 2.3: Clustering applied to 300hPa meridional wind field on 21 Jan 2007 (0600 UTC).
Black points represent filtered critical points. Red shapes represent minima clusters while
green shapes represent maxima clusters.

The details of the formulation of the similarity measure is given by [PMN20]. It
depends on the length of the shortest path between two local maxima/minima points
that lies on superlevel/sublevel sets of a given threshold value (2m/s and -2m/s in this
case) and the minimum/maximum value along the straight line connecting two local
maxima/minima.
The output of the clustering process is shown in the figure 2.3. The red shapes represent minima clusters and the green shapes represent maxima clusters. The clustering
algorithm in this case uses 150 as the diagonal element in the similarity matrix. We can
also see from the diagram that the zero-isocontour divides the wind field into separate
regions. Therefore, each maxima and minima cluster has a particular region associated
with it.
• Computing the association graph: The next step in the identification algorithm is
to identify the relevant associations between maxima clusters and minima clusters to
construct an association graph. This is an important step as the maxima clusters and
minima clusters identified earlier together form a wave packet. To identify these pairwise associations, we look for evidence for a shared boundary between pairs of maxima
and minima clusters. The boundary we consider for this method is the zero isocontour
of the wind field as any path from a maxima cluster to a minima cluster would certainly pass through this isocontour. Hence for each point on the zero isocontour, we
look for its closest local maximum and closest local minimum points. We then consider the maxima cluster and minima cluster to which these local maximum and local
minimum points belong, to be associated. A graph that stores all such associations
10

Figure 2.4: Association graph obtained for 300hPa meridional wind field on 21 Jan 2007
(0600 UTC).The blue points represent minima clusters and the red points represent maxima clusters; The solid purple line represents the zero-isocontour; The green line represents
associations between maxima and minima clusters.
between maxima and minima clusters is created and is called the association graph.
It is a bipartite graph between the maxima and minima clusters whose edges denote
associations between maxima clusters and minima clusters. In this graph, a maxima
cluster can have more than one minima cluster associated with it and vice versa.
Figure 2.4 illustrates the association graph. For display purposes, we use the point
with the largest magnitude of v for each cluster as a representative point for that
cluster. The blue points represent the corresponding minima clusters and the red
points represent the corresponding maxima clusters; The solid purple line represents
the zero-isocontour.
• Pruning the association graph: The association graph constructed in the last step may
contain several irrelevant edges, i.e, weak associations between maxima and minima
clusters that actually belong to different RWPs. Hence, it is necessary to prune the
association graph to make sure that individual RWPs are separate and retain only valid
associations. To prune the association graph, we assign two types of weights to each
edge in the graph: the scalar weight and the estimated gradient. The scalar weight is
based on the largest absolute value of v of the component nodes. We can use this to
identify the clusters that are part of an RWP by choosing a high threshold value for
the scalar weights. The estimated gradient on the other is a more physically motivated
measure. RWPs are usually separated by low curvature vorticity areas. The estimated
gradient is used to estimate the curvature vorticity linked to a (maxima cluster, minima cluster) pair. Based on our assumption we can use a high enough threshold for the
estimated gradient to further prune the association graph. Hence, these two measures
11

Figure 2.5: Scalar Pruned association graph obtained for 300hPa meridional wind field on
21 Jan 2007 (0600 UTC)

Figure 2.6: Edge Pruned association graph obtained for 300hPa meridional wind field on 21
Jan 2007 (0600 UTC)
together act as a good way to prune the association graph into separate connected
components each of which represents regions of RWP activity.
Figure 2.5 shows the association graph after the scalar pruning process. The scalar
threshold used in this case is 30m/s. Figure 2.6 shows the graph obtained post both
scalar and edge pruning steps. The threshold for estimated gradient used in this case is
0.02. As in the association graph, the blue points represent the corresponding minima
clusters and the red points represent the corresponding maxima clusters; The solid
black line represents the zero-isocontour.
• Extracting and displaying representative paths: The last step is to extract clear representative paths for each identified RWP from the pruned association graph. This step
is purely for visualisation purposes and all information regarding the local maxima
and local minima in each cluster is retained. We approach this step as an optimisation
problem. The pruned association graph consists of nodes that represent maxima and
minima clusters and weighted edges that represent associations and their strength be12

Figure 2.7: Final RWPs obtained for 300hPa meridional wind field on 21 Jan 2007 (0600
UTC)
tween two clusters. For the optimisation, we first compute all the hamiltonian paths
within each individual RWP region, i.e, simple paths passing through each node of the
graph exactly once. We then assign a path score to each of these paths by summing
the edge weights (sum of scalar weight and estimated gradient) of each of the edges
constituting the path. Finally, for each RWP, the path with the largest score is extracted and shown as the representative RWP.
Figure 2.7 shows the representative paths obtained for each identified RWP on 21
January 2007 (0600 UTC). Each node represents the local maxima or local minima
cluster that constitutes the particular RWP. Each node in this final graph contains all
the information of the critical points in that particular cluster. This information gives
important geometric properties of an identified RWP.

2.4

Modifications

In this section, we discuss the changes made in the implementation of the above algorithm.
• Identification of Critical Points: Critical points of the dataset were located using a
neighbourhood based computation in the previous implementation. This was done by
specifying a radius and locating a maximum and minimum for areas of this radius.
However, this method is dependent on the resolution of the input data and would need
revising of the radius parameter with varying resolution of the dataset. To overcome
this limitation, we have implemented a grid-based comparison method for determining
13

the critical points. In this implementation, we compare the scalar value at each grid
point to that of its neighbours to determine the local maxima and minima.
• Making use of Pyvista objects: We have used the PyVista module to store the data as
opposed to the previous implementation of the algorithm which mainly used vtk objects
for the same. PyVista is an assistance module for the Visualization Toolkit (VTK)
that uses NumPy and direct array access to interact with VTK in a different way. This
tool provides a Pythonic, well-documented interface for quickly prototyping, analysing,
and visualising geographically related datasets. Since all pyvista mesh objects are
subclasses of the corresponding vtk objects, these can be transparently used by vtk
functions. We use the pyvista Rectilinear Grid object to store the data. The Rectilinear
Grid can be created using the numpy arrays that represent latitude and longitude
using the function pyvista.RectilinearGrid(*args) (whose arguments are (lon,lat)
in this case). We can add the ”v” values and the local maxima or local minima values
identified by the addMaxData or addMinData functions respectively (both of which are
numpy arrays) to the Rectilinear Grid by using the point arrays attribute of PyVista
gridded objects. The array-like nature of PyVista objects makes it easier to use as
opposed to vtk objects. This implementation also gets rid of the paraview dependency
of the algorithm for visualisation and conversion of the dataset from NetCDF files
to vtk objects. Paraview is a platform for interactive data analysis and scientific
visualisation. In the current implementation, we use xarray, a python module, that is
designed specifically to work with NetCDF files for the conversion of NetCDF files into
numpy like arrays which can be then used by PyVista to create PyVista objects.
• Changes to the clustering code: As discussed earlier, the method uses the affinity propagation clustering algorithm to partition the identified local maxima and minima into
clusters. We choose this algorithm mainly as it supports different similarity measures
and also due to its capability to select the number of clusters on its own. In the affinity
propagation algorithm, the diagonal weights in the similarity matrix control the number of clusters chosen. If a value close to the minimum possible similarity is chosen,
the algorithm will produce a fewer number of clusters, while if a value greater than the
maximum possible similarity is chosen, the algorithm will produce a higher number
of clusters. We have mainly used this diagonal similarity parameter to optimise the
clustering process across several case studies. We chose the value 150 for the diagonal
similarity parameter as it gave the most intuitive clusters for all the case studies con14

sidered.
Figure 2.8 shows the output of the clustering algorithm for four different diagonal
weights for the wind field on 21 January 2007 (0600 UTC). As mentioned earlier, the
red shapes represent minima clusters and the green shapes represent maxima clusters.
The black points represent the identified critical points after filtration. As we can see
smaller diagonal weights such as 10 and 100 contains a relatively larger number of
single-point clusters while higher diagonal weights such as 400 produces fewer clusters
but also clusters spatially distant points together.

(a) Diagonal Weight = 10

(b) Diagonal Weight = 100

(c) Diagonal Weight = 150

(d) Diagonal Weight = 400

Figure 2.8: Clustering using different diagonal weights in the affinity propagation clustering
algorithm. Clusters are computed for meridional wind field at 300hPa (0600 UTC 21 January
2007)

• Changing the association and pruned association graph implementations: In the earlier
implementation, the association and pruned association graphs were stored as vtk
objects. In the modification implemented, all the graphs are stored as NetworkX graph
objects which are easier to handle than vtk objects. These NetworkX graph objects
consist of nodes that correspond to a maxima or minima cluster and edges between
these nodes, which also has weights associated with them. Each node in a graph stores
all the information about the constituent points of the corresponding cluster, such
as its cluster id, coordinates and scalar value. This information could be useful for
further analysis of the extent, amplitude, orientation etc. of the RWP components. A
representative point (the point whose absolute scalar value is maximum) is also chosen
15

for each cluster.
• Changing the representative paths graph computation: As we have seen earlier, after
the pruning step, we choose a representative path for each connected component for
visualisation purposes. Earlier, this was done by taking all the points in each cluster
into consideration. For each connected component, we would look at all the possible
paths. The nodes constituting these paths are maxima and minima clusters. All
possible combinations between the points in each constituent node of the path are
considered, and a path weight for each combination is calculated. These path weights
would then be used for choosing the optimal representative paths. In the modification
implemented, instead of looking at all possible combinations of the points in each
cluster and computing path weights for these, we compute the weight of a path using
the edge weights between the nodes that have been previously computed and used for
the pruning step. Hence we are avoiding the computation of path weight for each path
in the Cartesian product of the critical points in the constituent clusters which takes
O[nn ] time. Instead, we directly use the summation of the previously calculated edge
weights between the nodes and optimise it over all simple paths within each RWP.
This implementation uses far fewer steps as compared to the previous and hence is
computationally less complex. Since this step is only for visualisation purposes, we do
not lose any valuable information due to this simplification.

16

Chapter 3
Tracking of RWPs
In this chapter, we discuss the formulation and implementation of the tracking algorithm.

3.1

The Tracking problem

As discussed earlier, RWPs are essential in weather prediction since they have been linked
to downstream extreme weather occurrences as well as the development of forecast errors.
Tracking RWPs could also help us better understand how the large-scale flow regime is
changing. Hence we are interested in exploring an automated feature-based RWP tracking
approach similar to [SCC14b]. A feature-based method can substantially simplify the task
of efficiently investigating big datasets. Each domain has a unique set of key features. Since
these features appear in consecutive time steps, tracking them can improve visualisation and
present a useful analytical tool for studying the behaviour of these features. Furthermore,
tracking can aid in the automation of searches for events and interactions. It’s challenging
to keep track of features automatically because they’re always changing and interacting.
The first attempts of feature tracking [SSZC94] were presented, and the tracking was done
using approximations to features. In this work, we describe a new method for tracking that
uses a spatial overlap-based approach to correlate features from one dataset to features in a
subsequent dataset similar to the volume-based tracking approach presented in [SW96].
17

3.2

Methodology

3.2.1

Feature Identification

The first step of a feature-based tracking approach is feature selection. We have already
defined the method to identify and store RWPs for any dataset in a graphical form. The
regions encompassing each RWPs are the features that we aim to track across time. For
this, we first draw closed contours enclosing just one maxima or one minima cluster for each
cluster constituting the RWP. We then construct a convex hull encompassing all the regions
associated with each extrema cluster constituting the RWP. Hence the region associated
with each RWP in a timestep is represented using convex hull objects. All the grid points
within that object are attributed to the particular RWP. To construct the contours, we use
a factor α of the maximum absolute value in each RWP. To make further processing of these
feature geometries simpler, we rasterize the plot to obtain an image array with the input
geometries burned in. Rasterisation is the process of converting an image into a collection
of pixels, dots, or lines, that, when displayed together, form the image that was previously
represented by shapes. This is known as a raster image. Since the raster images obtained
are in the form of NumPy arrays, they can be easily used for all further computations.
Figure 3.1 shows how the raster image for the features identified for a particular timestep
(here, 0600 UTC 21 January 2007) looks like. The polygons in solid black lines in (a) represent the regions associated with each identified RWP in the timestep. The raster image (b)
has a dimension of 512 x 512. This is done as we require an image of dimension of the form
2n x 2n , where n is any natural number, for further processing. Furthermore, the features in
the raster image are contained within the original dimension of 90 x 360.

3.2.2

Feature Extraction

Now that we have identified the features to be tracked and stored them in an easily usable
form, we need to extract these features to effectively apply spatial overlap techniques. The
extraction technique should be broad enough to support a wide range of feature shapes while
yet being simple enough to track and quantify. For this purpose, we use quadtrees to extract
features in each timestep. Quadtrees are tree data structures in which every node except
18

(a) Identified features

(b) Raster image

Figure 3.1: (a) Regions of RWP activity (features) identified for 21 January 2007 (0600
UTC) (b) Rasterised image with features identified for 21 January 2007 (0600 UTC) burned
in

19

the root nodes has exactly four children. The quadtree data structure has the advantage of
having simple and efficient set operations (union, intersection, and difference calculations,
adjacency, membership testing, and transformations such as translation, rotation, and scaling). This efficiency can be attributed to the spatial indexing feature of the quadtrees. We
now look at how a quadtree is constructed.

Quadtree construction
A quadtree can be used to represent a 2n x 2n image where each pixel corresponds to a
feature or the background (denoted by the value 0). Hence, prior to using the quadtree
construction algorithm, we convert the raster image to an array of dimension 2n x 2n . This
is done by simply denoting the additional pixels required to be background pixels (of value
0) and would not alter the computation.
To construct the quadtree, we first create and insert the root node that represents the
entire region covered by the image (it consists of all the features present in that particular
timestep). Then the image is subdivided into four regions. If any of the subdivisions do not
entirely correspond to a single feature or the background, then it is again subdivided into 4.
This process of subdividing is continued recursively until all leaf nodes of the quadtree either
represent a region containing a single feature or a region containing only the background.
Each of the nodes contains information about all the features enclosed by the particular
region. It is constructed in the form of a directed graph where edges are directed from the
parent node to the daughter node. The quadtree is divided into different levels and the nodes
corresponding to each level denote regions of a particular size. Also, to facilitate spatial indexing, we index the daughter nodes of a node i as (4i +j) where j is a number between 1
and 4 (both included). This is useful as for two images of the same dimension, the nodes
with the same index denote the same region in the images, spatially.
It is apparent from this formulation that quadtrees save a lot of space while storing images. Rather than storing a large 2-D array of every pixel in the image, a quadtree can
record the same information at many divisive levels higher than the pixel-resolution-sized
cells we’d otherwise need. The height and resolution of the tree are limited by the pixel and
image sizes. The algorithm also returns the number of pixels corresponding to each feature
in the timestep.
20

Figure 3.2: Quadtree representation of an image of dimension 8x8[Mu8]

Figure 3.2 shows how an image of size 8x8 is represented by a quadtree. The black pixels in the image represent features and the white pixels represent the background. Similarly,
the white nodes in the quadtree represent a spatial region that is entirely part of the background and black nodes represent a region that is entirely constituted by feature pixels. Grey
nodes indicate a region which consist of both feature and background pixels.

3.2.3

Feature Tracking

Once we have defined and extracted the features for each timestep, we can define an algorithm
to track these features across time. Prior to that, we discuss a few evolutionary events that
occur in time-varying datasets.
• Continuation is an event in which a feature in the current timestep continues in the next
timestep. The feature may undergo rotation or translation and may even strengthen (
become bigger- grow) or weaken (become smaller) in size.
Figure 3.3 (a) shows an example of continuation of features from one timestep to
the next. An RWP region identified on 26 January 2007 (0600 UTC) overlaps with
an RWP region identified on 27 January 2007 (0600 UTC). The polygon in solid black
line represents the RWP region identified on 27 January 2007 and the polygon in the
dotted black line represents the RWP region identified on 26 January 2007.
• Creation is an event where a feature absent in the current timestep emerges in the next
21

timestep.
• Dissipation is an event where a feature present in the current timestep weakens and
becomes part of the background in the next timestep.
• Bifurcation is an event where a feature in the current timestep splits into two or more
features in the next timestep.
Figure 3.3 (b) shows an example of features that split in consecutive timesteps. An
RWP region identified on 21 January 2007 (0600 UTC) overlaps with two RWP regions
identified on 22 January 2007 (0600 UTC). The polygons in solid black line represent
the RWP regions identified on 22 January 2007 and the polygon in dotted black line
represents the RWP region identified on 21 January 2007.
• Amalgamation is an event where two or more features in the current timestep merge
into a single feature in the next timestep.
Figure 3.3 (c) shows an example of features that merge in consecutive timesteps. Two
RWP regions identified on 22 January 2007 (0600 UTC) overlaps with one RWP region
identified on 23 January 2007 (0600 UTC). The polygon in solid black line represents
the RWP region identified on 23 January 2007 and the polygons in dotted black line
represent the RWP regions identified on 22 January 2007.
We now discuss the algorithm to compute the spatial overlap between features in two consecutive time steps. The features in each timestep are stored in the form of quadtree graphs.
We make use of the spatial indexing feature of the quadtrees for this computation. Consider
two quadtrees for consecutive timesteps ti and ti+1 . For each index i, if it is part of both the
quadtrees, we compare the nodes at this index as follows: If one of the nodes is a background
node, we do not consider the descendants of this node in either tree in further iterations. If
both the nodes are feature nodes, the regions corresponding to the index overlap. The extent
of this overlap can be computed using the number of pixels the region consists of based on
the level (height) of the node in the tree. If both the nodes are mixed nodes (non-leaf nodes),
then we further compare the descendants of these nodes corresponding to the same indices.
If one of the nodes is a mixed node while the other is a feature node, we traverse the tree
rooted at the mixed node and identify the leaf nodes corresponding to individual features.
These leaf nodes would represent regions of overlap with the feature from the other timestep.
22

(a) Continuation of features in consecutive timesteps

(b) Features split in consecutive timesteps

(c) Features merge in consecutive timesteps

Figure 3.3: Evolutionary events that occur in time-varying datasets. Polygons in solid black
line represents RWP polygons identified in the current timestep and the polygons in dotted
black line represents RWP polygons identified in the previous timestep

23

The extent of the overlaps can be computed using the number of pixels the regions consist
of based on the level (height) of the nodes in the tree.

24

Chapter 4
Results and Conclusions

4.1

Feature Identification Calculations

As we previously discussed, feature identification is the first step in a feature-based tracking
pipeline. The method we implemented chooses regions of high amplitude meridional winds,
which encloses the RWPs identified previously, as features. For this we choose a fraction
of the absolute maximum scalar value in each identified RWP as the threshold for the construction of level sets. This ensures that the resulting polygons enclosing each RWP contains
only regions of significance and the tracking is not much affected by weaker regions (smaller
amplitude components) of the identified wave packets. Based on case studies, we have found
that using 0.5 as the fraction for computation of thresholds yields reasonable enclosing regions for the identified RWPs. Figure 4.1 shows bounding polygons calculated for 3 factors
- 0.25, 0.5 and 0.75 for RWPs identified on 20 January 2007. The bold black line represents
the representative wave packets for this timestep. The red shapes represent maxima clusters
and the green shapes represent minima clusters.

4.2

Tracking Graph Calculations

For tracking the regions (features) identified, our approach has been to match the regions in
consecutive timesteps based on their spatial overlap. We compute the overlap between fea25

(a) α=0.25

(b) α=0.5

(c) α=0.75

Figure 4.1: Regions identified for individual RWPs using different factors. The panel shows
meridional wind field at 300hPa (0600 UTC 20 January 2007). The bold black lines denote
the representative RWPs in that timestep and the polygons in the thin black lines denote
the identified feature
tures in two adjacent timesteps by performing an intersection operation using the quadtrees
that store the features for each time step. This operation gives the result in the form of
the number of pixels overlapped between each pair of overlapping features. This can be
represented in the form of a multipartite graph where each independent set of the vertices
of this graph represent a particular timestep and the nodes in this set represent the features
of this timestep. The edges of this graph would represent extend of overlap between the
respective features.
As we saw for the computation of the association graph in the identification pipeline, each
overlap detected by this algorithm may not be significant and represent the evolution of
RWPs across time. Hence we assign a measure of significance to each overlap detected. The
measure that we use for this implementation (similar to that used by [SW96]) is given below:
W (Ai , B i+1 ) =

Q(Ai , B i+1 )
max(QiA , Qi+1
B )

where A is a feature in timestep ti and B is a feature in timestep ti+1 ; Q(Ai , B i+1 ) denotes
the number of pixels that constitute the region of overlap between A and B; QiA represents
the total number of pixels that constitute the feature A and Qi+1
B represents the total number
of pixels that constitute the feature A.
26

Pruning the overlaps detected between features of consecutive timesteps based on this measure using an appropriate threshold value would help get rid of irrelevant overlaps. This
measure is also useful for the computation of bifurcations and amalgamations of features. In
these cases, we use the combinations instead of single features in the weight calculation. For
example, Let’s consider the case where a feature A in timestep ti overlaps with features B and
C in timestep ti+1 . We can compute the weight for all three cases- W (Ai , B i+1 ),W (Ai , C i+1 )
and W (Ai , (B ∪ C)i+1 ). The combination with the maximum weight is chosen as the track.
Since we use a quadtree representation for the features, computation of all the parameters
involved in weight calculation can be done simultaneously.
Using the above measure, we have computed the weights for all possible combinations of
overlapping features (including unions) and retained the combination with the maximum
weight. Furthermore, we could also prune these edges using a threshold for the edge weights
to ensure that no irrelevant tracks are being identified.
Figures 4.2 and 4.3 describe the evolution of a feature using the tracking method described
above. Here we have represented 4 timesteps in each figure. The tracking is done from 12
January 2007 to 18 January 2007 (Figure 4.2) and 16 January 2007 to 26 January 2007 (Figure 4.3) . The polygons using solid black lines represent a region of RWP activity (feature)
in the current timestep and the polygons using dotted black lines represent a region of RWP
activity (feature) in the previous timestep. Black points denote the centroids (representative
points) of the RWP polygons in the current timestep whereas red points denote the centroids
(representative points) of the RWP polygons in all the previous timestep starting from 12
January 2007 (for Figure 4.2) and 16 January 2007 (for Figure 4.3). Black solid lines represent the overlap between features of the current timestep and the preceding timestep. The
dotted black lines represent all the overlaps detected previously.
As we can see, although the algorithm captures RWP movement more or less accurately,
there are some anomalies. For example, from 15 January 2007 to 16 January 2007, the RWP
being tracked seems to have moved towards the west (as can be seen from the positions of the
polygons). However, this is contrary to what we expect due to the property of downstream
development of RWPs. The larger polygon on 16 January encompassing the polygon on 15
January could either be the result of the amplification of the weaker RWP on 15 January,

27

or it could be the result of an amalgamation of the RWP on 15 January with another RWP
propagating towards the west. We can also see that the RWP during the time interval from
16 January to 18 January 2007 increases in size towards the eastern end while the western
end is almost stationary. This could indicate a split of the initial RWP. However, this split
is not being captured by the tracking algorithm. Another anomaly is that in Figure 4.3,
we can see that a feature splits into two on 22 January, merges back into one feature on
23 January and splits again into two on 24 January. This should ideally have been a single
RWP.

4.3

Discussion and Conclusion

We have made several modifications in the identification algorithm for better computational
efficiency. These include: implementing an input dataset resolution-independent computation to locate local maxima and minima; Getting rid of paraview dependency by using simpler
alternatives for computation and visualisation such as Pyvista,Numpy,xarray,Matplotlib
etc; changing the clustering code so as to obtain more intuitive clusters; changes in the
graph implementations for easier handling and reduced computational complexity without
losing any phase information. This method also facilitates the identification of RWPs of any
orientation unlike previous attempts at identification.
We have also implemented an algorithm to track these RWPs. The quadtree representation of features in each timestep optimises the storage of these features. It also facilitates
simultaneous and easier computation of overlap between features and the size of these features. We also construct a tracking graph that describes the evolution of RWPs across time
which also captures splits and merges of features. The graph representation of tracking
makes it easy to analyse and also use filtering techniques.
Although conceptually thorough, it has been observed that the tracking algorithm sometimes produce counter intuitive results. Due to the complexity of RWP development, we
have not yet been able to get optimal RWP tracks using the tracking algorithm. We would
need further validation and modifications to obtain physically intuitive results for tracking.
Another limitation of this method is that due to 2D projection of geographic data that we
use, we cannot often identify and track every RWP. This could be overcome this by making
28

(a) 12 January 2007

(b) 14 January 2007

(c) 16 January 2007

(d) 18 January 2007

Figure 4.2: Regions identified for individual RWPs using different factors. The panel shows
meridional wind field at 300hPa (0600 UTC 12 January - 18 January 2007). Black points
represent centroids of features identified in the current timestep. Red points represent centroids of features identified in the previous timesteps. The solid black lines connecting red
and black points denote the movement of the RWP from the previous timestep to the current.
The dotted black lines connecting red and red points denote all the previous movements of
the RWP.
29

(a) 20 January 2007

(b) 22 January 2007

(c) 24 January 2007

(d) 26 January 2007

Figure 4.3: Regions identified for individual RWPs using different factors. The panel shows
meridional wind field at 300hPa (0600 UTC 20 January - 26 January 2007)

30

corrections for identification and tracking at the edges or using a spherical projection for
the dataset. Also, due to the formulation of features as polygons enclosing sublevel and
superlevel sets of a particular threshold value, features in the same timestep may overlap.
All the data used in this work can be downloaded from ECMWF. The identification and
tracking code will be available shortly open source.

31

32

Bibliography
[ABKC13] Heather M. Archambault, Lance F. Bosart, Daniel Keyser, and Jason M.
Cordeira. A climatological analysis of the extratropical flow response to recurving western north pacific tropical cyclones. Monthly Weather Review,
141(7):2325 – 2346, 2013.
[DNN13]

Harish Doraiswamy, Vijay Natarajan, and Ravi S. Nanjundiah. An exploration
framework to identify and track movement of cloud systems. IEEE Transactions
on Visualization and Computer Graphics, 19(12):2896–2905, 2013.

[FD07]

Brendan J. Frey and Delbert Dueck. Clustering by passing messages between
data points. Science, 315(5814):972–976, 2007.

[GDJ+ 11]

Ilona Glatt, Andreas Dornbrack, Sarah Jones, Julia Keller, O. Martius, Aurelia Muller, Dieter H. W. Peters, and Volkmar Wirth. Utility of hovmöller
diagrams to diagnose rossby wave trains. Tellus A: Dynamic Meteorology and
Oceanography, 63(5):991–1006, 2011.

[GFW18]

Paolo Ghinassi, Georgios Fragkoulidis, and Volkmar Wirth. Local finiteamplitude wave activity as a diagnostic for rossby wave packets. Monthly
Weather Review, 146(12):4099 – 4114, 2018.

[GV]

Federico Grazzini and Frédéric Vitart.
Atmospheric predictability and
rossby wave packets. Quarterly Journal of the Royal Meteorological Society,
141(692):2793–2802.

[Hak05]

Gregory J. Hakim. Vertical structure of midlatitude analysis and forecast errors.
Monthly Weather Review, 133(3):567 – 578, 2005.

[HLH+ 16]

C. Heine, H. Leitte, M. Hlawitschka, F. Iuricich, L. De Floriani, G. Scheuermann, H. Hagen, and C. Garth. A survey of topology-based methods in visualization. Computer Graphics Forum, 35(3):643–667, 2016.

[KHNH12] Jens Kasten, Ingrid Hotz, Bernd Noack, and Hans-Christian Hege. Vortex
Merge Graphs in Two-dimensional Unsteady Flow Fields. In Miriah Meyer and
33

Tino Weinkaufs, editors, EuroVis - Short Papers. The Eurographics Association,
2012.
[Mu8]

Wojciech Mula. Bitmap and its compressed quadtree representation, 2008.

[PMN20]

Karran Pandey, Joy Merwin Monteiro, and Vijay Natarajan. An integrated
geometric and topological approach for the identification and visual analysis of
rossby wave packets. Monthly Weather Review, 148(8):3139 – 3155, 2020.

[RKG+ 11] Jan Reininghaus, Natallia Kotava, David Günther, Jens Kasten, and Ingrid
Hotz. A scale space based persistence measure for critical points in 2d scalar
fields. IEEE Transactions on Visualization and Computer Graphics, 17:2045–52,
12 2011.
[RKWH12] Jan Reininghaus, Jens Kasten, Tino Weinkauf, and Ingrid Hotz. Efficient computation of combinatorial feature flow fields. IEEE Trans. Vis. Comput. Graph.,
18(9):1563–1573, September 2012.
[SCC14a]

Matthew B. Souders, Brian A. Colle, and Edmund K. M. Chang. The climatology and characteristics of rossby wave packets using a feature-based tracking
technique. Monthly Weather Review, 142(10):3528 – 3548, 2014.

[SCC14b]

Matthew B. Souders, Brian. A. Colle, and Edmund K. M. Chang. A description
and evaluation of an automated approach for feature-based tracking of rossby
wave packets. Monthly Weather Review, 142(10):3505 – 3527, 2014.

[SSZC94]

R. Samtaney, D. Silver, N. Zabusky, and J. Cao. Visualizing features and tracking their evolution. Computer, 27(7):20–27, 1994.

[SW96]

Deborah Silver and Xin Wang. Volume tracking. Proceedings of Seventh Annual
IEEE Visualization ’96, pages 157–164, 1996.

[VMN+ 19] Akash Anil Valsangkar, Joy Merwin Monteiro, Vidya Narayanan, Ingrid Hotz,
and Vijay Natarajan. An exploratory framework for cyclone identification and
tracking. IEEE Trans. Vis. Comput. Graph., 25(3):1460–1473, March 2019.
[WRCM18] Volkmar Wirth, Michael Riemer, Edmund K. M. Chang, and Olivia Martius.
Rossby wave packets on the midlatitude waveguide—a review. Monthly Weather
Review, 146(7):1965 – 2001, 2018.
[YMS+ 21]

Lin Yan, Talha Bin Masood, Raghavendra Sridharamurthy, Farhan Rasheed,
Vijay Natarajan, Ingrid Hotz, and Bei Wang. Scalar field comparison with
topological descriptors: Properties and applications for scientific visualization.
Computer Graphics Forum, 40(3):599–633, 2021.
34

[ZSP+ 03]

Aleksey V. Zimin, Istvan Szunyogh, D. J. Patil, Brian R. Hunt, and Edward
Ott. Extracting envelopes of rossby wave packets. Monthly Weather Review,
131(5):1011 – 1017, 2003.

35

