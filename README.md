# Tensorflow_api
Object detection code using tf api



Introduction

Steps

1a.Download TensorFlow Object Detection API repository from GitHub
Create a folder directly in C: and name it “tensorflow1”. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.

Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into the C:\tensorflow1 directory you just created. Rename “models-master” to just “models”.

Note: The TensorFlow models repository's code (which contains the object detection API) is continuously updated by the developers. Sometimes they make changes that break functionality with old versions of TensorFlow. It is always best to use the latest version of TensorFlow and download the latest models repository. If you are not using the latest version, clone or download the commit for the version you are using as listed in the table below.

2b. Download the ssd resnet50 model from TensorFlow's model
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo. Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy.


All files in \object_detection\images\train and \object_detection\images\test
The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
All files in \object_detection\training
All files in \object_detection\inference_graph





2e. Configure PYTHONPATH environment variable
A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. Do this by issuing the following commands (from any directory):

(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
(Note: Every time the "tensorflow1" virtual environment is exited, the PYTHONPATH variable is reset and needs to be set up again. You can use "echo %PYTHONPATH% to see if it has been set or not.)

2f. Compile Protobufs and run setup.py
Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters. Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API installation page does not work on Windows. Every .proto file in the \object_detection\protos directory must be called out individually by the command.

In the Anaconda Command Prompt, change directories to the \models\research directory:

(tensorflow1) C:\> cd C:\tensorflow1\models\research
Then copy and paste the following command into the command line and press Enter:

protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.

(Note: TensorFlow occassionally adds new .proto files to the \protos folder. If you get an error saying ImportError: cannot import name 'something_something_pb2' , you may need to update the protoc command to include the new .proto files.)

Finally, run the following commands from the C:\tensorflow1\models\research directory:

(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install

4. Generate Training Data
With the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model. This tutorial uses the xml_to_csv.py and generate_tfrecord.py scripts.

First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:


Next, open the generate_tfrecord.py file in a text editor where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file .




# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'apple':
        return 1
    elif row_label == 'banana':
        return 2
    elif row_label == 'orange':
        return 3
    else:
        None
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:

python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

--->Create Label Map and Configure Training
The last thing to do before training is to create a label map and edit the training configuration file.

----> Label map
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. (Make sure the file type is .pbtxt, not .txt !) In the text editor, copy or type in the label map in the format below (the example below is the label map for my Pinochle Deck Card Detector):

item {
  id: 1
  name: 'apple'
}

item {
  id: 2
  name: 'banana'
}

item {
  id: 3
  name: 'orange'
}


The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file,the labelmap.pbtxt file will look like:

item {
  id: 1
  name: 'apple'
}

item {
  id: 2
  name: 'banana'
}

item {
  id: 3
  name: 'orange'
}
---->Configure training
Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!

There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the config file.

Change num_classes to the number of different objects you want the classifier to detect.it would be num_classes : 3 .

Line 106. Change fine_tune_checkpoint to:

fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/ssd_v2_coco_2018_01_28/model.ckpt"
Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/train.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
Line 130. Change num_examples to the number of images you have in the \images\test directory.

Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/test.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

6. Run the Training
----->Go to workspace and copy python file from models/research/object_detection/model_main_tf2.py to workspace directory


From the \workspace directory, issue the following command to begin training:

python model_main_tf2.py --pipeline_config_path=D:/Ml_ref/Tensorflow/workspace/models/v1/pipeline.config --model_dir=D:/Ml_ref/Tensorflow/workspace/models/v1 --checkpoint_every_n=10 --num_workers=1 --alsologtostderr


Each step of training reports the loss. It will start high and get lower and lower as training progresses.  MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.

The checkpoint at the highest number of steps will be used to generate the frozen inference graph.

--->Export Inference Graph
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

python exporter_main_v2.py --input_type image_tensor --pipeline_config_path D:/Ml_ref/Tensorflow/workspace/models/v1/pipeline.config --trained_checkpoint_dir D:/Ml_ref/Tensorflow/workspace/models/v1/trained_check --output_directory D:/Ml_ref/Tensorflow

This creates a saved_model.pb file in the \saved_models\ folder. The .pb file contains the object detection classifier.

--->Trained Object Detection Classifier!
Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes. (I want to detect, so NUM_CLASSES = 3)

---->cd into the \workspace folder, and change the IMAGE_NAME variable in the new_test.py to match the file name of the picture.

python new_test.py

Generated a ouput with "output.jpg" in the workspace directory



