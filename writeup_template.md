# **Behavioral Cloning** 

## Submission Write up

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I used generator to process only a batch of images at a time, to reduce usage of memory exponentially
i. I used the flipping image technique to correct the left side and right side bias driving and to increase the volume of train and test data.
ii. Then I cropped the images from top 50 px and bottom 20 px to filter the unneccessary portion of the image
iii. Then I normalized the image data using lamda function
I used two models for this assignment.
1. Model derived from lenet model. 
    a. I have added convolution2D layer with 6 5x5 filters, with valid padding and elu activation as params
    b. Then, I did maxpooling to reduce the training time and also to make model to adapt to non linear solutions.
    c. I have added convolution2D layer with 6 5x5 filters, with valid padding and relu activation as params
    d. Then, I did maxpooling to reduce the training time and also to make model to adapt to non linear solutions.
    e. I have added convolution2D layer with 6 5x5 filters, with valid padding and relu activation as params
    f. Then, I did maxpooling to reduce the training time and also to make model to adapt to non linear solutions.
    g. Then, Flatten all the weights
    h. Then, added a Fully connected with 256 classes and relu as activation function
    i. Then, added a Fully connected with 64 classes and relu as activation function
    j. Then, added a Fully connected with 1 class and relu as activation function ( 1 class for steering angle)
2. Model derived from nvidia model.
    a. I have added convolution2D layer with 24 5x5 filters, with valid padding, 2x2 stride movement, weight regularizer and elu activation as params
    b. Then, I did dropout to reduce the training time and also to make model to adapt to non linear solutions.
    c. I have added convolution2D layer with 36 5x5 filters
    d. Then, I did dropout.
    e. I have added convolution2D layer with 48 3x3 filters
    f. Then, I did dropout.  
    e. I have added convolution2D layer with 64 3x3 filters
    f. Then, I did dropout.
    g. Then, Flatten all the weights
    h. Then, added a Fully connected with 2048 classes and elu as activation function
    i. Then, added a Fully connected with 64 classes and elu as activation function
    j. Then, added a Fully connected with 16 classes and elu as activation function
    h. Then, added a Fully connected with 1 class and elu as activation function ( 1 class for steering angle)
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, I have also used training data to recover from obstacles, like recover from hitting a tree or bridge etc.,
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that , my model will get adapt to both tracks

After training, I used drive.py to run car in autonomous mode. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and entered into a gravel road, which make it go into a dead end. to improve the driving behavior in these cases, I tried to train that particular portion of road couple of times, and trained to recover from the dead end. 
Then, I realized I need to train 4 classes - steering, throttle, brake and speed
I tried to train, but, failed to predict 4 values using the model.

At the end of the process, the vehicle is able to drive autonomously around the  first track without leaving the road. For second track, I did some manual driving to avoid car leaving the road some times

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 
i. I used generator to process only a batch of images at a time, to reduce usage of memory exponentially
ii. I used the flipping image technique to correct the left side and right side bias driving and to increase the volume of train and test data.
iii. Then I cropped the images from top 50 px and bottom 20 px to filter the unneccessary portion of the image
iv. Then I normalized the image data using lamda function
Model derived from nvidia model.
    a. I have added convolution2D layer with 24 5x5 filters, with valid padding, 2x2 stride movement, weight regularizer and elu                 activation as params
    b. Then, I did dropout to reduce the training time and also to make model to adapt to non linear solutions.
    c. I have added convolution2D layer with 36 5x5 filters
    d. Then, I did dropout.
    e. I have added convolution2D layer with 48 3x3 filters
    f. Then, I did dropout.  
    e. I have added convolution2D layer with 64 3x3 filters
    f. Then, I did dropout.
    g. Then, Flatten all the weights
    h. Then, added a Fully connected with 2048 classes and elu as activation function
    i. Then, added a Fully connected with 64 classes and elu as activation function
    j. Then, added a Fully connected with 16 classes and elu as activation function
    h. Then, added a Fully connected with 1 class and elu as activation function ( 1 class for steering angle)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

[![N|Solid](https://github.com/tkhgf/CarND-Behavioral-Cloning-P3-master/blob/master/examples/sample_1st_conv.jpg)]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:
[![N|Solid](https://github.com/tkhgf/CarND-Behavioral-Cloning-P3-master/blob/master/examples/flipped_sample_1st_conv.jpg)]

After the collection process, I had X number of data points. I then preprocessed this data by cropping
[![N|Solid](https://github.com/tkhgf/CarND-Behavioral-Cloning-P3-master/blob/master/examples/sample_cropped.jpg)]
Image dimension at this stage: 3 x 90 x 320
I finally randomly shuffled the data set and put Y% of the data into a validation set. 

After 1st convolution of 24 5x5 filters
[![N|Solid](https://github.com/tkhgf/CarND-Behavioral-Cloning-P3-master/blob/master/examples/sample_1st_conv.jpg)]
logits dimenision become 24 x 45 x 160 

After 2nd convolution of 36 5x5 filters
[![N|Solid](https://github.com/tkhgf/CarND-Behavioral-Cloning-P3-master/blob/master/examples/sample_2nd_conv.jpg)]
logits dimenision become 24 x 36 x 22 x 80

After 3rd convolution of 48 3 x 3 filters
[![N|Solid](https://github.com/tkhgf/CarND-Behavioral-Cloning-P3-master/blob/master/examples/sample_3rd_conv.jpg)]
logits dimenision become 24 x 36 x 48 x 12 x 41 

After 4th convolution of 64 3 x 3 filters
[![N|Solid](https://github.com/tkhgf/CarND-Behavioral-Cloning-P3-master/blob/master/examples/sample_4th_conv.jpg)]
logits dimenision become 24 x 36 x 48 x 64 x 7 x 21

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by my training results. I used an adam optimizer so that manually training the learning rate wasn't necessary.
