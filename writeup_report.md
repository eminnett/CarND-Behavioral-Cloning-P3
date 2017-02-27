# **behavioural Cloning**

## Writeup

---

**behavioural Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behaviour
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarise the results with a written report


[//]: # (Image References)

[image1]: ./examples/left_2017_02_26_18_56_09_850.jpg "Left Image"
[image2]: ./examples/center_2017_02_26_18_56_09_850.jpg "Centre Image"
[image3]: ./examples/right_2017_02_26_18_56_09_850.jpg "Right Image"
[image4]: ./examples/center_2017_02_26_10_00_58_291.jpg "Recovery Image 1"
[image5]: ./examples/center_2017_02_26_10_00_58_739.jpg "Recovery Image 2"
[image6]: ./examples/center_2017_02_26_10_00_59_189.jpg "Recovery Image 3"
[image7]: ./examples/center_2017_02_26_10_00_59_505.jpg "Recovery Image 4"
[image8]: ./examples/turn_left_2.jpg "Generated Image"
[image9]: ./examples/turn_right_2.jpg "Flipped Image"
[image10]: ./examples/generated_image.png "Generated Image"
[image11]: ./examples/resampled_image.png "Resampled Image"
[image12]: ./examples/cropped_image.png "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarising the results
* `video.mp4` showing the 'dashboard' view of the car while navigating the track autonomously at 30 mph (both counterclockwise and clockwise)
* `video_(screen_capture)_T1_CCW+CW_30mph_24fps.mp4` showing the screen view of the terminal and simulator while the car navigates the track autonomously (both counterclockwise and clockwise)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The two videos have been included in the project in case there are hardware / software conflicts.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The input of the model is a quarter size image (80x160 px instead of 160x320px). This resizing is performed using OpenCV before the images are fed through the network.

The first layer of the model crops the input to remove the top 25 rows and bottom 15 rows of pixels leaving the 40 rows of pixels that include the road in the image. (line 116 of model.py)

The remainder of the model is based on the CNN developed by Nvidia as written about in their paper, 'End to End Learning for Self-Driving Cars' (https://arxiv.org/pdf/1604.07316v1.pdf). I started with a copy of their network and then modified it to work with the smaller input image. The remaining parts of the Nvidia network are left in the code, but commented out.

Code was added to normalise the input of the network (line 118 of model.py), but I managed to train the model without this and found that the model performed worse in the simulator when I added it back to the model.

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48 (model.py lines 140-142), and 5 fully connected layers with progressively fewer nodes starting with 1164 and ending with a single output node. (model.py lines 151-160)

The model includes 'elu' (Exponential Linear Units) activations to introduce nonlinearity (a setting on every layer of the network from lines 140 to 158 of model.py). (Reference on ELU layers: https://arxiv.org/pdf/1511.07289v1.pdf)

#### 2. Attempts to reduce overfitting in the model

`model.py` contains dropout layers in order to reduce overfitting (model.py lines 153, 155, 157, and 159), but they were not utilised by the final model as they caused the model to underfit. Instead of dropout layers, I found decreasing the resolution of the input (and as a result decreasing the number of features) was a more successful way to get the model to perform well in the simulator.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested against 6 sample images and by running it through the simulator and ensuring that the vehicle could stay on the track.

I used the 6 sample images (2 left, 2 straight, and 2 right) not to define a conclusive test for the model, but more to judge whether a given model was overfitting or, more importantly, underfitting. If I could quickly find out whether a given model was simply predicting the same or very similar steering angles for these images, I could avoid the time it took to spin up the simulator and watch the car drive badly.

#### 3. Model parameter tuning

The model used an adam optimiser, so the learning rate was not tuned manually (model.py line 167).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of centre lane driving, recovering from the left and right sides of the road, smooth cornering, and driving both clockwise and counterclockwise around the track. I then added additional data to fine tune the driving behaviour once I had a model that worked reasonably well.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to copy an existing architecture and explore how manipulating the the input, add to the network and removing layers from the network impact its ability to predict the car's steering angle.

My first step was to use two different convolution neural network models similar to those developed by Comma AI (https://github.com/commaai/research) and Nvidia (https://arxiv.org/pdf/1604.07316v1.pdf). I thought these models might be appropriate places to start as both networks were developed to solve very similar problems to the one presented by this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with 20% set aside for validation. I found that many of my initial models underfit the data and simply returned the same steering angle for every input or all of the predictions were very similar to each other.

I was confused by this as I expected the network to overfit the data rather than underfit. In an attempt to diagnose the issue, I simplified the network to just a single convolution layer and two fully connected layers. I also decreased the complexity of the input by converting the images to a single grayscale channel. At this point, the model began to differentiate between left and right turns but was far from able to navigate the track.

At this point I compared the two networks, Comma AI and Nvidia, to get a sense as to which one would be the best to pursue for the project. After a bit of experimentation, I settled on the Nvidia network as I felt its having a larger number of layers would provide more opportunity to turn layers 'off and on' and find a model that worked well enough to fine tune.

In the end, I found that using RGB images but at a quarter resolution and with the top and bottom cropped along with a modified version of the Nvidia network worked best. The resizing and cropping results in an input of 40x160x3 or 19200 values compared to the 153600 values generated by the simulator. This is a reduction of 87.5%. This new input shape with a version of the Nvidia network with normalisation, dropouts, and the 4th and 5th convolution layers turned off resulted in the most consistent performance.

This was quite surprising as I would have expected the normalisation and dropout layers to improve performance rather than diminish it, but including these layers resulted in reasonably low training and validation error loss, but left the model unable to navigate around the track.

Even if it was a counter-intuitive, I decided to accept the unconventional structure of the network and accept the evidence of the performance of the autonomous car and its ability to navigate the track.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle struggled to navigate a few corners 'safely', but this was resolved by collecting additional training data to help the model perform appropriately in the problem areas.

At the end of the process, the vehicle was able to drive autonomously around the track both clockwise and counterclockwise without leaving the road at 30mph

#### 2. Final Model Architecture

The final model architecture (model.py lines 114-160) consists of a convolution neural network with the following layers and layer sizes

| Layer             | Input Shape  | Output Shape |
|:-----------------:|:------------:|:------------:|
| Crop              | (80, 160, 3) | (40, 160, 3) |
| Convolution (5x5) | (40, 160, 3) | (18, 78, 24) |
| Convolution (5x5) | (18, 78, 24) | (7, 37, 36)  |
| Convolution (5x5) | (7, 37, 36)  | (2, 17, 48)  |
| Flatten           | (2, 17, 48)  | (1632)       |
| Fully Connected   | (1632)       | (1164)       |
| Fully Connected   | (1164)       | (100)        |
| Fully Connected   | (100)        | (50)         |
| Fully Connected   | (50)         | (10)         |
| Fully Connected   | (10)         | (1)          |

#### 3. Creation of the Training Set & Training Process

To capture good driving behaviour, I first recorded three laps on the track driving carefully clockwise and another 3 laps driving counterclockwise.

I then recorded the vehicle recovering from the left side and right sides of the road back to centre so that the vehicle would learn to steer back to the centre of the track if it veered too close to the centre. I performed one lap clockwise and another counterclockwise weaving left and then weaving right and repeating.

These images show what a recovery looks like starting from the car riding on the edge of a curve and turning away from it. These images are 4 frames from a series of 35. In the sequence, there are 5 additional frames between each of these.

![recovery image 1][image4]
![recovery image 2][image5]
![recovery image 3][image6]
![recovery image 4][image7]

I then generated additional data where I carefully navigated the sharper corners of the track both clockwise and counterclockwise.

In total, this process resulted in 54,063 images (including those generated by the left, centre, and right cameras) and 18,021 steering angles.

These three images show the left, centre, and right camera views for a single 'frame'.

![left image][image1]
![centre image][image2]
![right image][image3]

The left, centre, and right images, along with flipped versions of each, were processed inside a generator function (the generator function is lines 41 through 103 of model.py). The images were converted to the RGB colour space (having been read in as BGR images by OpenCV) and resampled to shrink them to a quarter of their size (half both width and height). Steering angles for the left and right images were derived to be the centre steering angle +/- an offset value. The offset value of 0.1 was found to be adequate through a trial and error. Larger values tended to cause the car to weave left and right and often loose control (especially at higher speeds) while smaller values for the steering offset resulted in underfitting. The steering angles for the flipped images were simply the opposite than those for the original images.

These two images show an image generated by the simulator and the same frame flipped across its centre.

![generated image][image8]
![flipped image][image9]

These three images show an image generated by the simulator, the resampled version and the version of the image after it has been passed through the cropping layer of the model.

![generated image][image10]
![resampled image][image11]
![cropped image][image12]

I tried excluding images that described straight driving (a centre image with a steering angle within a certain threshold near 0), but found the model trained better when I included all 6 images for every 'frame' of the data.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I settled on 5 epochs as I found any more than that resulted in diminishing returns and impeded the speed at which I could iterate through the training / testing process. I used an adam optimiser so that manually training the learning rate wasn't necessary.

### Videos

To aid in the review of this project, I created two videos to illustrate the models performance when driving the autonomous car on track one of the simulator.

`video.mp4` shows the car driving from around the track one and half times counterclockwise (the direction the car faces when the simulator starts) the view is from the centre dashboard camera. There is a break in the footage while I manually drove the car through the off road area in order to turn the car around. The footage then shows the car completing one lap driving clockwise around the track. The video runs at 24 FPS with the car travelling 30 mph.

`video_(screen_capture)_T1_CCW+CW_30mph_24fps.mp4` shows the same 'run' as `video.mp4` but is a screen capture of the the terminal and simulator applications. This video includes the section of the run where the car was driven manually around the off road section of the track in order to turn the car around between completing counterclockwise and clockwise laps. This video is also 24 FPS with the car travelling 30 mph.
