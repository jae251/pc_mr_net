# Point Convolution and Map Regression Neural Net

In development

Neural net with some ideas for better object recognition
in point clouds:

+ Instead of using 3d convolution on voxels use point convolutions with n nearest points
+ For each point predict the relative position of the bounding box middle point

This avoids having to use any kind of pooling or fully connected layers, because
all computation can happen with convolution layers. Instead of predicting a set of objects,
the association of each point with an object is predicted.