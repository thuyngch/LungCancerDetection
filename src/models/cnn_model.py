"""
A conv net model using tflearn wrapper for tensorflow
"""

import tflearn
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import local_response_normalization

import numpy as np
import pandas as pd 
import tensorflow as tf 
import pickle, h5py
from src.models.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss


#------------------------------------------------------------------------------
#  CNNModel
#------------------------------------------------------------------------------
class CNNModel(object):
	def __init__(self):
		self.model = None

	def preprocessing(self):
		img_prep = ImagePreprocessing()
		img_prep.add_featurewise_zero_center()
		img_prep.add_featurewise_stdnorm()
		return img_prep

	def augmentation(self):
		img_aug = ImageAugmentation()
		img_aug.add_random_flip_leftright()
		img_aug.add_random_rotation(max_angle=25.)
		img_aug.add_random_blur(sigma_max=3.)
		return img_aug

	def input_layer(self, X_images, name):
		img_prep = self.preprocessing()
		img_aug = self.augmentation()
		return input_data(
			shape=[None, X_images.shape[1], X_images.shape[2], X_images.shape[3]],
			data_preprocessing=img_prep,
			data_augmentation=img_aug,
			name=name,
		)

	def se_layer(self, x, out_channels, name, activation_type='relu', ratio=0.25):
		squeeze = global_avg_pool(x, name=name+'_gap')
		excitation = fully_connected(squeeze, int(out_channels*ratio), activation=activation_type, name=name+'_fc1')
		excitation = fully_connected(excitation, out_channels, activation=activation_type, name=name+'_fc2')
		excitation = tf.nn.sigmoid(excitation)
		excitation = tf.reshape(excitation, [-1,1,1,out_channels])
		scale = x * excitation
		return scale

	def convolution_layer(self, x, num_filters, filter_size, name, activation_type='relu', regularizer=None, attention_ratio=0):
		x = conv_2d(x, num_filters, filter_size, activation=activation_type, regularizer=regularizer, name=name)
		if attention_ratio>0:
			x = self.se_layer(x, num_filters, name=name+'_se', activation_type=activation_type, ratio=attention_ratio)
		return x

	def max_pooling_layer(self, x, kernel_size, name):
		return max_pool_2d(x, kernel_size, name = name)

	def fully_connected_layer(self, x, num_units, activation_type, name):
		return fully_connected(x, num_units, activation=activation_type, name=name)

	def dropout_layer(self, x, name, prob=0.5):
		if (prob > 1) or (prob < 0):
			raise ValueError('Probability values should e between 0 and 1')
		return dropout(x, prob, name = name)

	def define_network(self, X_images, Y_targets, num_outputs=2, optimizer='adam', lr=1e-3,
						use_attention=False, use_triplet=False, triplet_hard_mining=False):

		x = self.input_layer(X_images, name='input')
		x = self.convolution_layer(x, 32, 5, 'conv1', 'relu', 'L2')
		x = self.max_pooling_layer(x, 2, 'mp1')
		x = self.convolution_layer(x, 64, 5, 'conv2', 'relu', 'L2')
		x = self.convolution_layer(x, 64, 3, 'conv3', 'relu', 'L2')
		x = self.max_pooling_layer(x, 2, 'mp2')
		x = self.fully_connected_layer(x, 512,'relu', 'fl1')
		x = self.dropout_layer(x, 'dp1', 0.5)
		placeholder = tf.placeholder(shape=[None, Y_targets.shape[1]], dtype=tf.float32, name="input_label")

		if use_triplet:
			x = self.fully_connected_layer(x, num_outputs, 'linear', 'fl2')
			loss = batch_hard_triplet_loss if triplet_hard_mining else batch_all_triplet_loss
			x = regression(
				x,
				optimizer=optimizer, 
				learning_rate=lr,
				loss=loss,
				placeholder=placeholder,
			)
		else:
			x = self.fully_connected_layer(x, num_outputs, 'softmax', 'fl2')
			x = regression(
				x,
				optimizer=optimizer, 
				learning_rate=lr,
				loss='categorical_crossentropy',
				placeholder=placeholder,
			)
		print("[{}] Build model with attention: {}".format(self.__class__.__name__, use_attention))
		print("[{}] Build model with TripletLoss: {}; HardMining: {}".format(self.__class__.__name__, use_triplet, triplet_hard_mining))
		return x


#------------------------------------------------------------------------------
#  Test bench
#------------------------------------------------------------------------------
if __name__ == "__main__":
	# Libraries
	import h5py

	# Build dataset
	h5f = h5py.File('src/data/train.h5', 'r')
	X_train_images = h5f['X']
	Y_train_labels = h5f['Y']

	h5f2 = h5py.File('src/data/val.h5', 'r')
	X_val_images = h5f2['X']
	Y_val_labels = h5f2['Y']

	# Build model
	convnet  = CNNModel()
	network = convnet.define_network(
		X_train_images, Y_train_labels, num_outputs=512,
		optimizer='adam', lr=1e-3, use_attention=True,
		use_triplet=False, triplet_hard_mining=False,
	)
	model = tflearn.DNN(
		network,
		max_checkpoints=10,
		tensorboard_verbose=1,
		checkpoint_path='ckpt/nodule3-classifier.ckpt',
	)
