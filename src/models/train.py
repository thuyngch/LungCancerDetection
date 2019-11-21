#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import numpy as np
import tflearn, h5py, os
from src.models.cnn_model import CNNModel


#------------------------------------------------------------------------------
#  Arguments
#------------------------------------------------------------------------------
train_data = "src/data/train.h5"
valid_data = "src/data/val.h5"
ckpt = "ckpt/attention0.5_softmax_bs8/nodule3-classifier.ckpt"
os.makedirs(os.path.dirname(ckpt), exist_ok=True)

lr = 5e-4
epochs = 100
batch_size = 8
attention_ratio = 0.5
use_triplet = False
triplet_hard_mining = False


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
if __name__ == "__main__":

	# Load HDF5 dataset
	h5f = h5py.File(train_data, 'r')
	X_train_images = h5f['X']
	Y_train_labels = h5f['Y']

	h5f2 = h5py.File(valid_data, 'r')
	X_val_images = h5f2['X']
	Y_val_labels = h5f2['Y']

	# Model definition
	convnet  = CNNModel()
	network = convnet.define_network(
		X_train_images, Y_train_labels, num_outputs=2, optimizer='adam', lr=lr,
		attention_ratio=attention_ratio, use_triplet=use_triplet, triplet_hard_mining=triplet_hard_mining,
	)
	model = tflearn.DNN(network)

	# Training and validating
	model.fit(
		X_train_images,
		Y_train_labels,
		n_epoch=epochs,
		shuffle=True,
		validation_set=(X_val_images, Y_val_labels),
		show_metric=True,
		batch_size=batch_size,
		snapshot_epoch=True,
		run_id='nodule3-classifier',
	)

	# Save checkpoint
	model.save(ckpt)
	h5f.close()
	h5f2.close()
