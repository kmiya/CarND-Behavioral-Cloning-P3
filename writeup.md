# Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around a track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image0]: ./images/network.png "NVIDIA's CNN Network"
[image1]: ./images/fail_wo_bn.png "Fail to drive a track"
[image2]: ./images/loss.png "Training/Validation loss"
[image3]: ./images/center_camera.jpg "Center Camera"
[image4]: ./images/left_camera.jpg "Left Camera"
[image5]: ./images/right_camera.jpg "Right Camera"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---

## Requirements

### Required Files

> Are all required files submitted?

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model_kb.h5` containing a trained convolution neural network
* `video_kb.mp4` is a recorded video of autonomous driving with `model_kb.h5`

Also includes additional files for comparison:

* `model_kb_bn.h5` containing a trained convolution neural network with Batch Normalization
* `video_kb_bn.mp4` is a recorded video of autonomous driving with `model_kb_bn.h5`

The above models are trained using a dataset captured using a keyboard. On the other hand, the following models are trained using two datasets captured using a keyboard and a mouse.

* `model_kb_mouse.h5` containing a trained convolution neural network
* `video_kb_mouse.mp4` is a recorded video of autonomous driving with `model_kb_mouse.h5`
* `model_kb_mouse_bn.h5` containing a trained convolution neural network with Batch Normalization
* `video_kb_mouse_bn.mp4` is a recorded video of autonomous driving with `model_kb_mouse_bn.h5`

### Quality of Code

#### 1. Is the code functional?

Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing

```sh
python drive.py model_bn.h5
```

Please be aware of the software dependencies and their versions described in `README.md`.

#### 2. Is the code usable and readable?

You can see a Python generator is used in `model.py` lines 31-49 (function `generator()`).

### Model Architecture and Training Strategy

#### 1. Has an appropriate model architecture been employed for the task?

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 64-78). The model is based on the NVIDIA's model (see [Bojarski et al. 2016](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)).

![NVIDIA's CNN Network][image0]

I've added Batch Normalization layers to the NVIDIA's model to reduce internal covariate shift (see [Ioffe & Szegedy 2015](http://proceedings.mlr.press/v37/ioffe15.pdf) for details), followed by ReLU layers to introduce nonlinearity. The input image is converted from RGB to YUV color space in the model by a Keras Lambda layer (line 62) and normalized by the first Batch Normalization layer (line 63).

#### 2. Has an attempt been made to reduce overfitting of the model?

The model contains Batch Normalization layers in order to reduce overfitting. I also stopped training with just one epoch for early stopping.

The model was trained and validated on different data sets to ensure that the model was not overfitting (lines 109-119). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Have the model parameters been tuned appropriately?

The model used an adam optimizer, so the learning rate was not tuned manually (line 108).

The number of epoch is 1 for early stopping, the batch size is 32 (lines 101-102).

#### 4. Is the training data chosen appropriately?

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The training data includes 5,880 x 3 = 17,640 images and its flipped images as augmented data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Document

#### 1. Solution Design Approach

First, I attempted to use the NVIDIA's model with normalized RGB images. But it was difficult to train with my training dataset, the car in the simulator was unable to drive around a track.

Next, I added Batch Normalization layers to the model. After training the model again, the car could drive around the track, but seemed to unstable to drive.

Finally, I added a lambda layer to the model to convert the color space of input images from RGB to YUV. It enabled the model to drive the car around the track stably. See `video_bn.mp4`.

#### 2. Final Model Architecture

The final model is described in the section above. The following is a summary of the model:

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 160, 320, 3)       0
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
lambda_1 (Lambda)            (None, 90, 320, 3)        0
batch_normalization_1 (Batch (None, 90, 320, 3)        12
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824
batch_normalization_2 (Batch (None, 43, 158, 24)       96
activation_1 (Activation)    (None, 43, 158, 24)       0
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636
batch_normalization_3 (Batch (None, 20, 77, 36)        144
activation_2 (Activation)    (None, 20, 77, 36)        0
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248
batch_normalization_4 (Batch (None, 8, 37, 48)         192
activation_3 (Activation)    (None, 8, 37, 48)         0
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712
batch_normalization_5 (Batch (None, 6, 35, 64)         256
activation_4 (Activation)    (None, 6, 35, 64)         0
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928
batch_normalization_6 (Batch (None, 4, 33, 64)         256
activation_5 (Activation)    (None, 4, 33, 64)         0
flatten_1 (Flatten)          (None, 8448)              0
dense_1 (Dense)              (None, 100)               844900
batch_normalization_7 (Batch (None, 100)               400
activation_6 (Activation)    (None, 100)               0
dense_2 (Dense)              (None, 50)                5050
batch_normalization_8 (Batch (None, 50)                200
activation_7 (Activation)    (None, 50)                0
dense_3 (Dense)              (None, 10)                510
batch_normalization_9 (Batch (None, 10)                40
activation_8 (Activation)    (None, 10)                0
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 983,415
Trainable params: 982,617
Non-trainable params: 798
```

During experiments, Batch Normalization plays an import role if the input images do _NOT_ be converted to YUV from RGB. However, when using the color space conversion I could not find out the effectiveness of the Batch Normalization. It decreased the training loss earlier but the impact for the validation loss seemed to be small in my project.

![Training / Validation loss][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving using a __keyboard__. Here is an example image of center lane driving:

![Center lane driving][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center using a __mouse__ so that the vehicle would learn how to recover to center of the road.

To augment the data, I added flipped images and angles to the data set. I also added the left and right camera images with the factor `0.2` to adjust the angles (lines 45-46).

![Left camera][image4]
![Right camera][image5]

After the collection process, I had 7,351 x 3 = 22,053 camera images and angles with flipped ones as augmented data.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The ideal number of epochs was 1, determined by experiments. Recall the training-validation loss curves above. Increasing the number of epochs causes overfitting of the model.

### Simulation

#### Performance Comparison

In the viewpoint of _smoothness_ of driving around Course one, the best model would be `model_kb.h5`, which was trained using the keyboard dataset only and does not include Batch Normalization layers. The keyboard dataset contains records of three laps on Course one using center lane driving, while the mouse dataset contains records of vehicle recovering from left and right sides to center. The records of recovering seem to make the model robust, but also seem to harm smoothness of driving. Batch Normalization also seems to make the model robust, but it seems to  make the model (or angles) more sensitive.
