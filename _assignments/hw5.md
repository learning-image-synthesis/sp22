---
type: assignment
date: 2021-04-19T4:00:00-5:00
title: 'Assignment #5 - GAN Photo Editing'
thumbnail: /static_files/assignments/hw5/thumb.gif
attachment: /static_files/assignments/hw5/hw5_starter.tar.gz
due_event:
    type: due
    date: 2021-04-30T23:59:00-5:00
    description: 'Assignment #5 due'
mathjax: true
hide_from_announcments: true
---

$$
\DeclareMathOperator{\argmin}{arg min}
\newcommand{\L}{\mathcal{L}}
\newcommand{\Latent}{\tilde{\mathbb{Z}}}
\newcommand{\R}{\mathbb{R}}
$$

{% include image.html url="/static_files/assignments/hw5/teaser.gif" %}
An example of grumpy cat outputs generated from sketch inputs using this assignment's output.

## Introduction
In this assignment, you will implement a few different techniques that require you to manipulate images on the manifold of natural images. First, we will invert a pre-trained generator to find a latent variable that closely reconstructs the given real image. In the second part of the assignment, we will take a hand-drawn sketch and generate an image that fits the sketch accordingly.

We have provided the starter [code](/static_files/assignments/starter.tar) and data and model file [here](https://drive.google.com/file/d/17IY0N7fKffdUl0MIsEx9fJ2FlyC64hN1/view?usp=sharing). You can try each problem with vanilla gan (in `vanilla/`) or a StyleGAN (in `stylegan`).

## Setup

This assignment is a little bit more picky about dependencies than the previous ones. You need to run the following command in a fresh virtualenv with a recent Python version:

`pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.

Furthermore, please install PyTorch (at least 1.7.1) with your CUDA version as 11.

All the code you need to write is in `main.py`.

## Part 1: Inverting the Generator [30 pts]
For the first part of the assignment, you will solve an optimization problem to reconstruct the image from a particular latent code. As we've discussed in class, natural images lie on a low-dimensional manifold. We choose to consider the output manifold of a trained generator as close to the natural image manifold. So, we can set up the following nonconvex optimization problem:

For some choice of loss \\(\L\\) and trained generator \\(G\\) and a given  real image \\(x\\), we can write

$$ z^* = \argmin_{z} \L(G(z), x).$$

Here, the only thing left undefined is the loss function. One theme of this course is that the standard Lp losses do not work well for image synthesis tasks. So we recommend you try out the Lp losses as well as some combination of the perceptual (content) losses from the previous assignment. As this is a nonconvex optimization problem where we can access gradients, we can attempt to solve it with any first-order or quasi-Newton optimization method. One issue here is that these optimizations can be both unstable and slow. Try running the optimization from many random seeds and taking a stable solution with the lowest loss as your final output.

### Implementation Details
* Fill out the `forward` function in the `Criterion` class. You'll need to implement each of the losses as well as a way of combining them. Feel free to add whatever arguments you want to accomplish this and properly configure your class. You may need to pass your input image into the loss to make this work. Feel free to include code from previous assignments. Do this in a way that works whether a mask is included or not.
* You also need to implement `sample_noise` -- this is obviously easy for the vanilla gan, but you should implement the sampling procedure for StyleGAN. Feel free to use the Grumpy Cat models you have trained in HW3.
* Next, implement the optimization step. We have included a different implementation of LBFGS as this one includes the line search. Feel free to experiment with other PyTorch optimizers such as SGD and Adam. You should implement the optimization function call in a general fashion so that you can reuse it.
* Finally, implement the whole functionality in `project` so you can run the inversion code.


### Deliverables
{% include image.html url="/static_files/assignments/hw5/interpolation.gif" align="left" width=200 %}


Show some example outputs of your image reconstruction efforts using (1) various combinations of the losses, (2) different generative models, and (3) different latent space (latent code, w space, and w+ space).  Give comments on why the various outputs look how they do. Which combination gives you the best result and how fast your method performs. 

## Part 2: Interpolate your Cats [10 pts]
Now that we have a technique for inverting the cat images, we can do arithmetic with the latent vectors we have just found. One simple example is interpolating through images via a convex combination of their inverses. More precisely, given images \\(x_1\\) and \\(x_2\\), compute \\(z_1 = G^{-1}(x_1), z_2 = G^{-1}(x_2)\\). Then we can combine the latent images for some \\(\theta \in (0, 1)\\) by \\(z' = \theta z_1 + (1 - \theta) z_2\\) and generate it via \\(x' = G(z')\\). Choose a discretization of \\((0, 1)\\) to interpolate your image pair.

Experiment with different generative models and different latent space (latent code z, w space, and w+ space)

### Implementation
* Implement the interpolation step in `interpolate` where you project, interpolate, and reconstruct the images and save them in image_list so that you can render a gif of the images smoothly transitioning.

### Deliverables

Show a few interpolations between grumpy cats. Comment on the quality of the images between the cats and how the interpolation proceeds visually.

## Part 3: Scribble to Image [40 Points]
Next, we would like to constrain our image in some way while having it look realistic. This constraint could be color scribble constraints as we initially tackle this problem, but could be many other things as well. We will initially develop this method in general and then talk about color scribble constraints in particular.  To generate an image subject to constraints, we solve a penalized nonconvex optimization problem. We'll assume the constraints are of the form \\(\{f_i(x) = v_i\}\\) for some scalar-valued functions \\(f_i\\) and scalar values \\(v_i\\).

Written in a form that includes our trained generator \\(G\\), this soft-constrained optimization problem is

$$z^* = \argmin_{z} \sum_i ||f_i(G(z)) - v_i||^2.$$

__Color Scribble Constraints:__
Given a user color scribble, we would like GAN to fill in the details. Say we have a hand-drawn scribble image \\(s \in \R^d\\) with a corresponding mask \\(m \in {0, 1}^d\\). Then for each pixel in the mask, we can add a constraint that the corresponding pixel in the generated image must be equal to the sketch, which might look like \\(m_i x_i = m_i s_i\\).

Since our color scribble constraints are all elementwise, we can reduce the above equation under our constraints to

$$z^* = \argmin_z ||M * G(z) - M * S||^2,$$

where \\(*\\) is the Hadamard product, \\(M\\) is the mask, and \\(S\\) is the sketch

### Implementation Details

* Implement the code for synthesizing images from drawings to realistic ones using the optimization procedure above in `draw`.
* You can use [this website](https://sketch.io/sketchpad/) to generate simple color scribble images of cats in your browser.
* We've provided here a color palette of colors which typically show up in grumpy cats along with their hex codes. You might find better results by using these common colors.
{% include image.html url="/static_files/assignments/hw5/colormap.png" %}


### Deliverables

Draw some cats and see what your model can come up with! Experiment with sparser and denser sketches and the use of color. Show us a handful of example outputs along with your commentary on what seems to have happened and why.

## Bells & Whistles (Extra Points)
Max of **15** points from the bells and whistles.
- Implement additional types of constraints. (3pts each): e.g., sketch/shape constraint and warping constraints mentioned in the iGAN paper, or texture constraint using a style loss. 
- Train a neural network to approximate the inverse generator (4pts) for faster inversion and use the inverted latent code to initialize your optimization problem (1 additional point).
- Develop a cool user interface and record a UI demo (4 pts). Write a cool front end for your optimization backend. 
- Experiment with high-res models of Grumpy Cat (2 pts) [data and pretrained weight from here](https://drive.google.com/file/d/1p9SlAZ_lwtewEM-UU6GvYEdQWTV1K-_g/view?usp=sharing) or other datasets (e.g., faces, pokemon) (2pts) 
- Other brilliant ideas you come up with. (up to 5pts)


## Further Resources
- [Generative Visual Manipulation on the Natural Image Manifold](https://arxiv.org/pdf/1609.03552.pdf)
- [Neural Photo Editing with Introspective Adversarial Networks](https://arxiv.org/abs/1609.07093)
- [Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?](https://arxiv.org/abs/1904.03189)
- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)
- [GAN Inversion: A Survey](https://arxiv.org/abs/2101.05278)

__Authors__:
This assignment is created by Jun-Yan Zhu, Viraj Mehta, and Yufei Ye.
