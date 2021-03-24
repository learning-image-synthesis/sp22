---
type: assignment
date: 2021-03-29T4:00:00-5:00
title: 'Assignment #4 - Cats in Style'
thumbnail: /static_files/assignments/hw4/teaser.png
attachment: /static_files/assignments/hw4/hw4_starter.tar.gz
due_event:
    type: due
    date: 2021-04-12T23:59:00-5:00
    description: 'Assignment #4 due'
mathjax: true
hide_from_announcments: true
---


{% include image.html url="/static_files/assignments/hw4/teaser.jpeg" %}
(Left: Jun-Yan's cat. Right: [todo?])

## Introduction
In this assignment, you will implement neural style transfer which resembles specific content in certain artistic style. For example, generate cat images in Ukiyo-e style. The algorithm takes in a content image, a style image, and another input image. The input image is optimized to match the previous two target images in content and style distance space. 

In the first part of the assignment, you will start from a random noise and optimize it in content space. It helps you get familiar with the general idea of optimizing pixels with respect to certain losses.  In the second part of the assignment, you will ignore content for a while and only optimize to generate textures. This builds some intuitive connection between style-space distance and gram matrix.  Lastly, we combine all of these pieces together to perform neural style transfer.   

We have provided the starter code and test images [here](/static_files/assignments/hw4/hw4_starter.tar.gz). 

## Part 1: Content Reconstruction [30 points]
For the first part of the assignment, you will implement content-space loss and optimize a random noise with respect to the content loss only.


__Content Loss:__ The content loss is a metric function that measures the content distance between two images at certain individual layer. Denote the Lth-layer feature of input image X as $f_X^L$ and that of target content image as $f_C^L$. The content loss is defined as squared L2-distance of these two features $\|f_X^L - f_C^L\|^2_2$.

Implement content loss in the code:
```angular2html
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        # todo: your implementation
        return input
```

__Feature extractor and Loss Insertion:__ Of course when $L$ equals 0, content-loss is just a L2 pixel loss, which does not represent content. So the content loss is actually in the feature space. To extract feature, a [VGG-19]() net pretrained on ImageNet is used. The pre-trained VGG-19 net consists of 5 convolution blocks (conv1-conv5) and each block serves as a feature extractor at different abstract level. We only optimize the conv5 feature with respect to content loss in this assignment. 

The pretrained VGG-19 can be directly imported from `torchvision.models` module and the loading utility has been provided in the starter code. Write your code to append content loss to the end of specific layers (will be ablated soon) in order to optimize.  

__Optimization:__ In contrast with assignment 3 where we optimize the parameters of a neural network, in assignment 4 we fix the neural network and optimize the pixel values of the input image. Here we use a quasi-newton optimizer `LBFGS` to optimize the image `optimizer = optim.LBFGS([input_img.requires_grad_()])`. The optimizer involves reevaluate your function multiple times so rather than a simple `loss.backward()`, we need to specify a hook  `closure` that performs 1) clear the gradient, 2) compute loss and gradient 3) return the loss.

Please complete the closure function in `run_optimization`:
```angular2html
    while run_num[0] < num_steps:

        def closure():
            # correct the values of updated input image
            # todo: your implementation
            return loss

        optimizer.step(closure)
```

__Experiment__: 
2. Report the effect of optimizing content loss at different layers. Choose your favorite one (specify it in the website) and: [15 points]
1. Take two random noise as two input images, optimize them only with content loss. Please include your results on the website and compare each other with the content image. [15 points]

## Part 2: Texture Synthesis [30 points]
Now let us implement style-space loss in this part. 

__Style loss:__ How do we measure the distance of the styles of two images? In the course we discussed that Gram matrix is used as a style measurement. Gram matrix is the correlation of two vectors on every dimension. Specifically, denote the k-th dimension of the Lth-layer feature of an image as $f^L_k$ in shape of $(N, K, H*W)$. Then the gram matrix is $G = f^L_k (f^L_k)^T$ in shape of (N, K, K).  The idea is that two of the gram matrix of our optimized and predicted feature should be as close as possible.  

Please implement the style loss and gram matrix in the code:
```angular2html
def gram_matrix(activations):
    # todo
    return gram 

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        # todo
        return output
```

__Applying loss:__ Similar to part one, please insert the implemented style loss to the desired layers. Here we optimize features from all conv layers: conv1 till conv5.  

__Optimization:__ If you write `run_optimization` smartly, lots of code can be reused by setting function argument as 0/1.  Please modify those parameters accordingly in order to only optimize style loss.  

__Experiment:__
1. Report the effect of optimizing texture loss at different layers. Use one of the configuration; specify it in the website and: [15 points]
1. Take two random noise as two input image, optimize them only with content loss. Please include your results on the website and compare these two synthesized textures. [15 points]

## Part 3: Style Transfer [40 points]
Finally, it is time to put pieces together!

__Applying Losses:__ The building blocks are almost ready. You need to insert both content and style loss to some certain layers. We would use configuration of conv5 for content and conv1-5 for style. 

__Experiment:__
1. Tune the hyper-parameters until you are satisfied. Pay special attention to whether your gram matrix is normalized over feature pixels or not. It will result in different hyper-parameters by an order of 4-5. Please briefly describe your implementation details in the website.  [15 points]
2. Please report at least 2x2 grid of results that are optimized from two content images mixing with two style images accordingly. (Remember to also include content and style images therefore the grid is actually 3x3) [15 points]
3. Try style transfer on some of your favorite images. [10 points] 

## What you need to submit
* Three code files and any necessary files to run your code: `run.py`, `style_and_content.py`, `utils.py`.
* A website submitted like the previous homeworks following the instructions [here](/assignments/hw0/) containing samples of content reconstruction, texture synthesis, and style transfer.


## Bells & Whistles (Extra Points)
Max of **10** points from the bells and whistles.
- Stylize your grump cats or poisson blended images from the previous homework. (2pts)
- Apply style transfer to a video. How do you handle the temporal axis? (4 pts)
- In the assignment a pretrained vgg-19 is used to extract features. Use your own feedforward network and describe the improvement you observed (10 pts)
- Add some spatial control by masking out certain regions(4 pts)


## Further Resources
- [Texture Synthesis Using Convolutional Neural
  Networks, Gatys et al., 2015](https://arxiv.org/pdf/1505.07376.pdf)
- [A Neural Algorithm of Artistic Style, Gatys et al., 2015
  ](https://arxiv.org/pdf/1508.06576.pdf)  

__Acknowledgement__:
The assignment is credit to Pytorch tutorial [neural transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).
