---
type: assignment
date: 2021-04-19T4:00:00-5:00
title: 'Assignment #5 - GAN Photo Editing'
thumbnail: /static_files/assignments/hw5/teaser.png
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

{% include image.html url="/static_files/assignments/hw5/teaser.png" %}
An example of grumpy cat outputs generated from sketch inputs using this assignment's output.

## Introduction
In this assignment, you will implement a few different techniques that require you to manipulate images on the manifold of natural images. First, we'll `invert' a trained generator to find a latent variable which generates an image very similar to one given. In the second part of the assigment, we'll take a sketch that you input and generate an image that fits the sketch from our latent space.

We have provided the starter [code](/static_files/assignments/starter.tar) and data and model file [here](https://drive.google.com/file/d/161V3oaL_SvV4qlxBs-kjl9XJlHcF3V4k/view?usp=sharing). You can try each problem with a vanilla gan (in `vanilla/`) or a StyleGAN (in `stylegan`).

## Setup

This assignment is a little bit more picky about dependencies than the previous ones. You need to run the following command in a fresh virtualenv with a recent Python version:

`pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.

Furthermore you need to make sure as you also install PyTorch that its version is at least 1.7.1 and your major CUDA version is 11.

All the code you need to write is in `main.py`.

## Part 1: Inverting the Generator [30 pts]
For the first part of the assignment, you'll solve an optimization problem to reconstruct the image from a particular latent code. As we've discussed in class, natural images lie on a manifold in image space. We choose to consider the output manifold of a trained generator as close to the natural image manifold. So, we can set up the following nonconvex optimization problem:

For some choice of loss \\(\L\\) and trained generator \\(G\\) and given some real image \\(x\\), we can write

$$ z^* = \argmin_{z} \L(G(z), x).$$

Here, the only thing left undefined is the loss function. One theme of this course is that the standard Lp losses aren't great for image tasks. So we recommend you try out the Lp losses as well as some combination of the perceptual (style and content) losses from the previous assigment. As this is a nonconvex optimization problem where we have access to gradients, we can attempt to solve it with any first-order optimization method. One issue here is that these optimizations can be unstable. Try running the optimization from many random seeds and taking a stable solution with low loss as the one you output.

### Implementation
* Fill out the `forward` function in the `Criterion` class. You'll need to implement each of the losses as well as a way of combining them. Feel free to add whatever arguments you want to accomplish this and properly configure your class. You may need to pass your input image into the loss to make this work. Feel free to include code from previous assignments. Do this in a way that works whether a mask is included or not.
* You'll also need to implement `sample_noise` -- this is obviously easy for the vanilla gan but you should implement the sampling procedure for StyleGAN as well.
* Next, implement the optimization step. We've included a different implementation of LBFGS as this one includes line search for step size. You should implement this in a general fashion so that you can reuse it.
* Finally, implement the whole functionality in `project` so you can run the inversion code.


### Deliverables
{% include image.html url="/static_files/assignments/hw5/interpolation.gif" align="left" width=200 %}


Show some example outputs of your image reconstruction efforts using various combinations of the losses. Give comments on why the various outputs look how they do.

## Part 2: Interpolate your Cats [10 pts]
Now that we have a technique for inverting the cat images, we can do arithmetic with the latent vectors we've found. One simple example is interpolating through images via a convex combination of their inverses. More precisely, given images \\(x_1\\) and \\(x_2\\), compute \\(z_1 = G^{-1}(x_1), z_2 = G^{-1}(x_2)\\). Then we can combine the latent images for some \\(\theta \in (0, 1)\\) by \\(z' = \theta z_1 + (1 - \theta) z_2\\) and generate it via \\(x' = G(z')\\). Choose a discretization of \\((0, 1)\\) to interpolate your image pair.

### Implementation
* Implement the interpolation step in `interpolate` where you project, interpolate, and reconstructe the images and save them in image_list so that you can render a gif of the images smoothly transitioning.

### Deliverables

Show a few interpolations between grumpy cats. Comment on the quality of the images between the cats and how the interpolation proceeds visually.

## Part 3: Sketch to Image [40 Points]
Next, we'd like to constrain our generated image in some way while having it look realistic. This constraints could be sketch constraints as we initially tackle in this problem, but could be many other things as well. We'll initially develop this method in general and then talk about sketch constraints in particular. In order to generate an image subject to constraints, we solve a penalized nonconvex optimization problem. We'll assume the constraints are of the form \\(\{f_i(x) = v_i\}\\) for some scalar-valued functions \\(f_i\\) and scalar values \\(v_i\\).

Written in a form that includes our trained generator \\(G\\), this soft-constrained optimization problem is

$$z^* = \argmin_{z} \sum_i ||f_i(G(z)) - v_i||^2.$$

__Sketch Constraints:__
Here, we allow for an initial sketch which will be painted in by the GAN. Say we have a sketch image \\(s \in \R^d\\) with a corresponding mask \\(m \in {0, 1}^d\\). Then for each pixel in the mask, we can add a constraint that the corresponding pixel in the generated image must be equal to the sketch, which might look like \\(m_i x_i = m_i s_i\\).

Since our sketch constraints are all elementwise, we can reduce the above equation under our constraints to

$$z^* = \argmin_z ||M * G(z) - M * S||^2,$$

where \\(*\\) is the Hadamard product, \\(M\\) is the mask, and \\(S\\) is the sketch

### Implementation

* Implement the code for synthesizing images from drawings to realistic ones using the optimization procedure above in `draw`.
* You can use [this website](https://sketch.io/sketchpad/) to generate simple sketch images of cats in your browser.
* We've provided here a color palette of colors which typically show up in grumpy cats along with their hex codes. You might find better results by using these common colors. The hex codes corresponding to these are (from left to right): #24201d, #d2cdc7, #5a372e, #745e37, #636562, #896c5f, #9f8a68, #909391, #af958c, #1cb843.
{% include image.html url="/static_files/assignments/hw5/color_palette.png" %}


### Deliverables

Sketch some cats and see what your model can come up with! Experiment with sparser and denser sketches and the use of color. Show us a handful of example outputs along with your commentary on what seems to have happened and why.

## Bells & Whistles (Extra Points)
Max of **10** points from the bells and whistles.
- Use gradients from the discriminator to push your generated images to look even more realistic (2pts). See equation 5 of [this paper](https://arxiv.org/pdf/1609.03552.pdf) for details.
- Implement addtional types of constraints. (3pts) The same paper gives you the ability to use HOG features to make the visual cues more approximate and structural rather than exact color.
- Train a neural network to approximate the inverse generator (2pts) for faster inversion and use the inverted latent code to initialize your optimization problem (1 additional point).
- Other brilliant ideas you come up with. (5pts)


## Further Resources
- [Generative Visual Manipulation on the Natural Image Manifold](https://arxiv.org/pdf/1609.03552.pdf)

__Authors__:
This assigment was written by Jun-Yan Zhu, Viraj Mehta, and Yufei Ye
