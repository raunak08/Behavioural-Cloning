import numpy as np
import csv
import cv2
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

def process_image(path):
    """
    Performs pre-processing on the images obtained from simulator
    Arguments:
        path--path to the directory of images
    Returns:
        image--processed image
    """
    # Read Image
    image = ndimage.imread(path)
    # Crop Image from top and bottom
    image = image[60:140,:,:]
    # Resize the Image to match the input size of the model
    image = cv2.resize(image,(64,64),cv2.INTER_AREA)
    return image

lines = []
images = []
steer_angles = []

with open('../Trial_4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    # Defining correction factor for steering angles of left and right images
    correction = 0.2
    steer_center = float(line[3])
    steer_left = steer_center + correction
    steer_right = steer_center - correction
    
    source_path = '../Trial_4/IMG/'
    center_path = source_path + line[0].split('\\')[-1]
    left_path = source_path + line[1].split('\\')[-1]
    right_path = source_path + line[2].split('\\')[-1]
    
    # Pre-processing of images
    center_image = process_image(center_path)
    left_image = process_image(left_path)
    right_image = process_image(right_path)
    
    images.append(center_image)
    images.append(left_image)
    images.append(right_image)
    
    steer_angles.append(steer_center)
    steer_angles.append(steer_left)
    steer_angles.append(steer_right)
   
X_train = np.array(images)
Y_train = np.array(steer_angles)

# Defining model
model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(64,64,3)))
model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3)))
model.add(ELU())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), input_shape=(32,32,64)))
model.add(ELU())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), input_shape=(16,16,128)))
model.add(ELU())
model.add(MaxPooling2D((2, 2)))
model.add(ELU())
model.add(Flatten())
model.add(Dense(128))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,Y_train, validation_split=0.2, shuffle=True)

model.save('../CarND-Behavioral-Cloning-P3/model_4.h5')
exit()