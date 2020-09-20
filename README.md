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

```
C:\> conda create -n tensorflow1 pip python=3.6
```

Then we will activate this virtual environment.

```
C:\> activate tensorflow1
```

```
(tensorflow1) C:\>python -m pip install --upgrade pip
```

Now we will install TensorFlow

```
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow
```

After this, we will install the necessary packages of python:

```
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
```

After this, we will Configure the PYTHONPATH environment variable. PYTHONPATH is an environment variable. The PYTHONPATH variable has a value that is a string with a list of directories that Python should add to our directory.

```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
 Next, we need to compile the Protobuf scripts, which TensorFlow uses to configure parameters for our model and for training it. This generates the name_pb2.py file in the \object detection\protos folder from every name.proto file. But for this first, we will change the directory.
 
 ```
 (tensorflow1) C:\> cd C:\tensorflow1\models\research
 ```
 
 And then run the following command
 
 
 ```
 protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
 ```
 
 Now finally, we will run the following to commands from the C:\tensorflow1\models\research.  The build command is used for putting all the files to install into the build directory.
 
 The TensorFlow Object Detection API is now fully configured to use pre-trained object detection models or train new models.
 
 ### Step 3: Customize particular files according to our dataset.
 
In this step, we will customize and tweak a few files in order to successfully run this objection detection task to identify facemask specifically. 
 
At first we will edit the file named xml_to_csv.py. From line number 17 to 20 we will change int(member[4][0].text) to int(member[5][0].text)

Now in the training folder (C:\tensorflow1\models\research\object_detection\training), we will delete all the pre-existing files and create a new file named labelmap.pbtxt. We will insert the following script in the labelmap file.

```
item {
  id: 1
  name: 'with_mask'
}

item {
  id: 2
  name: 'without_mask'
}

item {
  id: 3
  name: 'mask_weared_incorrect'
}
```

Next, we will convert the xml into csv by using below code:

```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```

This will create our CSV files that are train_labels.csv and test_labels.csv

Now, we will edit our generate_tfrecord.py file. we will replace the code with the following script

```
def class_text_to_int(row_label):
    if row_label == 'with_mask':
        return 1
    elif row_label == 'without_mask':
        return 2
    elif row_label == 'mask_weared_incorrect':
        return 3
    else:
        None
```

After editing it we will run the commands below to generate the TFRecord files that are train.record and test.record

```
(tensorflow1) C:\tensorflow1\models\research\object_detection>python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record

(tensorflow1) C:\tensorflow1\models\research\object_detection>python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

Now we are done with step 3 and will perform the complete step 4 into google collaboratory file Facemask_Detection_using_Faster_rcnn.ipynb. 

### Step 4: Test the model on Anaconda Command Prompt.

Once we have done training the model, we will copy all the files from the folder named  Final_Weights_Obtained saved on the google drive to our inference_graph folder stored in the object_detection folder.

After that, we will edit the Object_detection_image.py file and make the following changes.

  1. Change the name of the directory containing the object detection module we are using i.e. MODEL_NAME = 'inference_graph.' IMAGE_NAME = 't1.jpg'
  
  2. Add another line after line number 115 cv2.imwrite('output1.jpg', image)

After making the following changes, we will run the python file in the Anaconda command prompt. Hence, we have successfully trained and tested the facemask detection model using faster_rcnn architecture.

Cheers!!!


