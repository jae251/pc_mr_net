# Point Convolution and Map Regression Neural Net


Neural net with some ideas for better object recognition
in point clouds:

+ Instead of using 3d convolution over space convolve over each point
+ For each point predict the relative position of the associated bounding box middle point

This avoids having to use any kind of pooling or fully connected layers, because
all computation can happen with convolution layers. Instead of predicting a set of objects,
the association of each point with an object is predicted.

The lidar data is used in an image format, which enables similar architecture to image recognition networks.
However the first layer works more akin to a graph convolution (one could ofc say an image convolution is just
a special type of graph convolution): Given an adjacency distance (similar to image convolution kernel size)
the relative position to the adjacent vertices are used in a weighted sum. X,Y and Z coordinates
are used as channel informations.

The convolution layers are structured in modules similar to Inception Layers, just a bit smaller
to be more friendly to GPUs with smaller VRAM.

Things to come:
+ Visualization of the actual output of the neural net
+ More metriks for output quality
+ Colab implementation for cloud training
+ Use of sparse convolution for better performance
+ Experiments with different layer structures / parameters 