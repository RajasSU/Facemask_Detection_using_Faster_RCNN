# Facemask_Detection_using_Faster_RCNN
Technologies used: Python, Pandas, TensorFlow, Keras, OpenCV.

## Problem Description
This GitHub repository focuses on implementing Faster-RCNN on a Facemask dataset using TensorFlow’s Object Detection API.
The dataset that we will use is available on Kaggle as Face Mask Detection (https://www.kaggle.com/andrewmvd/face-mask-detection). With this dataset, it is possible to construct an object detection model to identify people wearing masks, not wearing them, or inappropriately wearing them. This dataset contains 853 images belonging to the 3 classes and their bounding boxes in the PASCAL VOC format. The classes are:
  1. With mask
  2. Without mask
  3. Mask wore incorrectly.
To do object detection to identify if people in the image are wearing a facemask, we will follow specific steps to obtain a fully trained model for object detection
  1. Set our TensorFlow1 Directory.
  2. Create an Anaconda Virtual Environment.
  3. Customize particular files according to our dataset.
  4. Train the Model on Google Colab.
  5. Test the model on Anaconda Command Prompt.
 
Test the model on Anaconda Command Prompt.
So before starting these steps, please make sure that you have installed the Anaconda Navigator (https://www.anaconda.com/products/individual#download-section) because we will be primarily using the Anaconda virtual environment to build out the Facemask Detection model. 

## Steps to be followed:

### Step 1: Set our TensorFlow1 Directory.
We will create a new folder directly in local disk C: and name it as “tensorflow1,” and then we will download the TensorFlow Object Detection API repository from GitHub (https://github.com/tensorflow/models). This “tensorflow1” will be our primary working directory, which will contain all the necessary files for our object detection framework.
After downloading the whole repository, we will extract it to our tensorflow1 folder and rename the “models-master” folder to “models.”

After that, we will download the EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10. We will copy all the folders and files from this repository and replace them in our object detection folder: C:\tensorflow1\models\research\object_detection. 

### Step 2: Create an Anaconda Virtual Environment.
Check the Anaconda Prompt utility from the Start menu in Windows, right-click on it and click “Run as Administrator.” 
If you are asked by Windows if you want to allow it to make improvements to your device, please press Yes.
In the Anaconda prompt, create a new virtual environment named “tensorflow1.”
```C:\> conda create -n tensorflow1 pip python=3.6```
Then we will activate this virtual environment.
```C:\> activate tensorflow1
(tensorflow1) C:\>python -m pip install --upgrade pip```
