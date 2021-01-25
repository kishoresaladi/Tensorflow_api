# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:43:58 2021

@author: kishore saladi
"""

#import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#
#detection_graph = tf.Graph()
#with tf.Session(graph=tf.Graph()) as sess:
#    tf.saved_model.loader.load(
#        sess, [tf.saved_model.tag_constants.SERVING], "D:/Ml_ref/Tensorflow/workspace/models/saved_model/")
    
    
import tensorflow as tf
import cv2
import numpy as np


import sys

# This is needed since the notebook is stored in the object_detection folder.


# Import utilites

sys.path.append("D:/Ml_ref/Tensorflow/models/research/object_detection")
from utils import visualization_utils as vis_util

#tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('D:/Ml_ref/Tensorflow/workspace/models/saved_model/')


image_path = "apple_1.jpg"
image_np = cv2.imread(image_path)
input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)
image_np_with_detections = image_np.copy()
category_index = {
    1: {'id': 1, 'name': 'apple'},
    2: {'id': 2, 'name': 'banana'},
    3: {'id': 3, 'name': 'orange'},
    }
#item {
#    name: "apple",
#    id: 1,
#    display_name: "apple"
#}
#item {
#    name: "banana",
#    id: 2,
#    display_name: "banana"
#}
#item {
#    name: "orange",
#    id: 3,
#    display_name: "orange"
#}

print(detections)

#detections['detection_boxes'][0].numpy(),detections['detection_classes'][0].numpy().astype(np.int32),detections['detection_scores'][0].numpy()
        
vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0,line_thickness=8)

cv2.imwrite("output.jpg",image_np_with_detections)
#plt.subplot(2, 1, i+1)
#plt.imshow(image_np_with_detections)


#   
#image_tensor = 'image_tensor:0'
#model_dir="D:\Ml_ref\Tensorflow\workspace\models\v1\trained_check"
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#import os
#checkpoint_path = os.path.join(model_dir, "ckpt-15")
#
## List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
#print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='')
#
## List contents of v0 tensor.
## Example output: tensor_name:  v0 [[[[  9.27958265e-02   7.40226209e-02   4.52989563e-02   3.15700471e-02
#print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='v0')
#
## List contents of v1 tensor.
#print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='v1')


#x=[n.name for n in detection_graph.as_graph_def().node]
