# bird-classification
Image classification using a neural network. This is a short documentation of my process.  
A video with a short demo and some more explanations can be found [here](https://www.youtube.com/watch?v=0YdsuQivxTU).

# Objective
Given an image of a bird, classify its species out of the 555 possible options. The dataset consisted of around 38000 training and 10000 testing images, all of birds. This project is my work for this [Kaggle competition](https://www.kaggle.com/c/birds21wi/).

My personal goals were to:
- See how people actually code with neural networks in practice
- Learn some Python and Pytorch
- Get experience with a doing a lot of self-learning

# Previous Work
I used some code snippets from this [tutorial](https://www.pluralsight.com/guides/image-classification-with-pytorch)
for getting the input data into the right format for the neural network and testing the network. I also consulted the Pytorch tutorials from class for the [simple neural network](https://colab.research.google.com/drive/1CYD8uaxc_J5xmkJWT3cKnaF4cnUASWJP?usp=sharing) and the [simple convolutional network](https://colab.research.google.com/drive/1aedcC_6-2j2Jz0BySbJgTSyWfrpmtarI). For my neural network I used a pretrained model ([ResNet34](https://pytorch.org/hub/pytorch_vision_resnet/)).

# Setup
Before I could start, I needed to familiarize myself with the tools available. I went through some tutorials on using Kaggle, which is where I got the data, and Google Colab, which I used as my coding environment because it provides free GPU usage. I also spent a decent amount of time Python and Pytorch tutorials before starting because I didn't have any experience and learn better with some structure rather than just going in and trying things out.

# Preparing data
After getting the data from Kaggle to Colab, I had to get the images in the right format for the neural network. This involved cropping the images to the same size and ensuring all images were in the same colorspace. I also further split up the training images into 80% training and 20% validation data so that I could get continual feedback on both testing and training accuracies without having to submit results to Kaggle every time. Finally, I tried out various data augmentation methods to prevent overfitting like normalization, random cropping, and random horizontal flipping.

At this point, my code was running very slowly because my program was reading the image data from Google Drive. Uploading the training and test data into the Colab VM disk and reading from there solved this issue.

# Training
I started by trying out a simple neural network and a simple convolutional neural network modelled after the examples from the tutorials from class. I wanted to start very simply so that I could understand every part of the neural network and then go on from there. The bird dataset is much harder than the datasets used in class, so understandably both of these models gave very poor results.

Next, I tried using a pretrained ResNet18 model with Adam optimizer, which gave much better results of around 40% test accuracy after finding the proper learning rate. My model was overfitting, so I tried out data augmentation as well as other techniques like dropout and weight decay. The data augmentation helped the most, but the model was still overfitting pretty severely.

At this point, I found this helpful [article](https://towardsdatascience.com/why-adamw-matters-736223f31b5d) that showed me that the choice of optimizer was important too. Basically, I learned that Adam is a popular optimizer because it converges faster than SGD, but it can be prone to overfitting and weight decay isn’t effective when using it. Since the bird training dataset has so many species (classes), there aren’t actually that many images per species. This means it's easier for the model to overfit because the quantity of training data is small. I tried out AdamW, which aims to fix the overfitting issue, as well as SGD.

![image](https://user-images.githubusercontent.com/31548288/110995384-25741c00-832f-11eb-90eb-e701cf9f3145.png) ![image](https://user-images.githubusercontent.com/31548288/110995405-2ad16680-832f-11eb-9132-fe80f04bfb08.png)

I didn’t notice a big difference between AdamW and Adam, but it was cool to see the characteristics described in the article in practice. You can see that the model using SGD has less overfitting, but would also require many more epochs of training to achieve the same level of accuracy as Adam.

Using SGD with momentum ended up being the best option. At this point, I tried increasing image input size from 128x128 to 256x256, using ResNet34 (which has more layers), and using a learning schedule. All of these helped my model get to its highest testing accuracy.

# Results
![image](https://user-images.githubusercontent.com/31548288/111003124-68d48780-833b-11eb-8909-f1ea90519082.png)  
After I finding the best parameters and other options that produced the best results, I retrained the model using the full training dataset (instead of splitting it into training and validation data). Then, I submitted the model predictions to Kaggle, getting a final testing accuracy of around 81%.

# Conclusion
While I initially struggled a lot because I’m used to more structured assignments and was unfamiliar with all of the tools I needed to use, I’m satisfied with the result of this project. I feel like I both completed the assignment and achieved my personal goals. Throughout this project, I read through a lot of documentation, articles, and academic papers. It was cool to be able to read about certain concepts and to be able to immediately apply them in my code.

I think my approach is relatively straightforward and has many aspects similar to what we learned about in class. However, in the process of experimenting with what combination worked the best, I learned about a lot of new concepts (like optimizer types or nesterov momentum) and got to try them out in my code, which was a valuable experience. If I had more time (and perhaps a higher GPU usage quota), then I would further adjust my model. I was satisfied after getting the test accuracy around 80%, but it does look like the model is still overfitting. I also might try experimenting with adjusting the internal ResNet layers.
