# CS128H-Project
This is Shaan Doshi and Kevin Huang's CS128H Final Project
<br>Group Name: ShaanKevinNN
<br>Net IDs: shaand3, kh47

Introduction:
<br>The project will involve building a neural network.
We are coding a neural network from scratch using no libraries. We are using a tree-based structure to implement forward propagation. We are then using a genetic algorithm to train our neural network on a dataset to find racial disparities in housing in certain areas. With this genetic algorithm, we can implement parallelism to speed up our processing of different generations. Our goal of the project is to accurately identify disparities through an accurately trained custom neural network. We chose this project because we've always been interested in machine learning and AI, but have always seen it used with libraries in just a couple of lines of code. Actually coding a neural network from scratch allows us to learn how every step of the process works, allowing a fully customizable network that we can apply to an interesting application like racial disparity.

Technical Overview:
<br>There will be 3 main Neural network classes: a Connectors class that connects nodes to each other with a certain weight and bias, a Node class that holds values and pushes them forward through the Connectors, and a Perceptron class that contains Nodes and Connectors in a structure so that forward propagation occurs. There will also be a main class that loads in the dataset, splits the data into training and testing, runs a parallelized genetic algorithm, then determines the final accuracy based on testing data.
<br>By Checkpoint 1, we hope to have the structure of the three neural network classes done to have a functioning forward propagation.
<br>By Checkpoint 2, we want to implement our parallelized genetic algorithm on a dataset.

Challenges:
<br>One challenge is that the genetic algorithm isn't accurate enough to classify the data, since we are not implementing convolutions or backpropagation.
Another challenge is getting the genetic algorithm to work parallelized.

References:
<br>One reference we are looking at is 3Blue1Brown's series on how neural networks work.
<br>https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=inWqAUjbBf3JnYgD
