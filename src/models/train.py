"""
Trains a CNN model using tflearn wrapper for tensorflow
"""

import numpy as np
import tflearn, h5py
from src.models.cnn_model import CNNModel 


# Load HDF5 dataset
h5f = h5py.File('src/data/train.h5', 'r')
X_train_images = h5f['X']
Y_train_labels = h5f['Y']


h5f2 = h5py.File('src/data/val.h5', 'r')
X_val_images = h5f2['X']
Y_val_labels = h5f2['Y']


## Model definition
convnet  = CNNModel()
network = convnet.define_network(X_train_images, num_outputs=2)

model = tflearn.DNN(
	network,
	tensorboard_verbose=0,
	checkpoint_path='ckpt/nodule3-classifier.tfl.ckpt',
)

model.fit(
	X_train_images,
	Y_train_labels,
	n_epoch=50,
	shuffle=True,
	validation_set=(X_val_images, Y_val_labels),
	show_metric=True,
	batch_size=512,
	snapshot_epoch=True,
	run_id='nodule3-classifier',
)

model.save("ckpt/nodule3-classifier.tfl")
print("Network trained and saved as ckpt/nodule2-classifier.tfl!")

h5f.close()
h5f2.close()
