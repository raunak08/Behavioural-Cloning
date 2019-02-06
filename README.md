# **Project: Behavioral Cloning** 

### In this project, a virtual car will be trained by learning human driving. This is done by utilizing Simulator, that records the data for human driving. This data can then be used to train the Deep Learning model to drive the car through the defined track.

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./IMG/center_2019_01_08_18_38_51_060.jpg "Right Turn"
[image2]: ./IMG/center_2019_01_10_07_39_08_408.jpg "Left Turn"
[image3]: ./IMG/left_2019_01_10_07_37_39_679.jpg "Left Camera Image"
[image4]: ./IMG/center_2019_01_10_07_37_39_679.jpg "Center Camera Image"
[image5]: ./IMG/right_2019_01_10_07_37_39_679.jpg "Right Camera Image"

---
* The github includes all required files and can be used to run the simulator in autonomous mode
Following are the files:
  * model.py containing the script to create and train the model (if required, changes to model can be made here!) 
  * drive.py for driving the car in autonomous mode
  * model.h5 containing a trained convolution neural network 
  * README.md summarizing the results (This file!)

* Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
  ```sh
  python drive.py model.h5
  ```
* The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


## Using simulator to record data
* The simulator provides to mode to play with, training mode and autonomous mode.
* Training mode is used to collect the data and autonomous mode is used to let the car run on the trained model.
* During training mode, the data collected is in images and a driving log, where driving log is a in csv format.
* Images recorded are Left, Center and Right view of the front side of the vehicle.
* Driving log consists of path to these images, steering angle, throttle, break, speed.
* By using this recorded data we can train our model.

## Model Architecture and Training Strategy

### 1. Model architecture

| Layer (type)                    | Output Shape      |                     
|---------------------------------|-------------------|
| Lambda                          | (64, 64, 3)       |         
| Convolution2D (3x3, 64 filters) | (64, 64, 64)      |
| Activation (ELU)                |   (64, 64, 64)    |
| MaxPooling2D (2,2 filter)       | (32, 32, 64)      |
| Convolution2D (3x3, 128 filters)|  (32, 32, 128)    |
| Activation (ELU)                | (32, 32, 128)     |
| MaxPooling2D (2,2 filter)       | (16, 16, 128)     |
| Convolution2D (3x3, 256 filters)| (16, 16, 256)     |
| Activation (ELU)                | (16, 16, 256)     |
| MaxPooling2D (2,2 filter)       | (8, 8, 256)       |
| Activation (ELU)                | (8, 8, 256)       |
| Flatten                         | (128)             |
| Activation (ELU)                | (128)             |
| Flatten                         | (1)               |

### 2. Solution Design Approach
The model used for this solution was inspired from the [DeepGaze](https://github.com/mpatacchiola/deepgaze) Head Pose Estimation Architecture. There the model is used to estimate the Yaw, Pitch and Roll angles of the head from Head images. The same model with neural style transfer approach has been utilised.

### 3. Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded two laps on track one with original lap direction and the other in opposite direction of the lap. Here is an example both the images of driving:

![alt text][image1]
![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center. These images show what a left, center and right camera images would look like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

After the collection process, I had 7935 number of data points. I then preprocessed this data by normalizing the data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over fitting or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### 4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road. For training data, I have utilised track in both, front and opposite direction, as discussed above. Thus collecting enough data for maximum left and right turns. Apart from center images, I have also utilised left and right images by adding corresponding correction factor (0.2) to steering angles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80-20%). 

### 5. Attempts to reduce overfitting in the model
The model was trained and validated on different data sets to ensure that the model was not overfitting. This data was obtained by recording the training data from simulator in two fashion: one, in regular format where there are more left turns, second, the data was recorded by using track in reverse direction. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 6. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

## Results and improvements
* The model performed decent enough, that it could drive autonomously without hitting the sides and also without crossing over the side lines of the road. Thus the car could drive harmlessly even if there was a living entity inside it.
* There are still certain performance improvements for this model. As you can see in the video, the car doesnot drive smoothly on straight tracks, it swings minutely left and right. The data can be increased by further augmenting it, and doing some image proceesing on it. Also, if it was required by the car to complete the track in very less time, then this model would probably crash, as it is tested only for certain maximum speed.
* A well trained and implemented model from Nvidia for same simulator can be used and further improve the performance by doing above mentioned activities.

