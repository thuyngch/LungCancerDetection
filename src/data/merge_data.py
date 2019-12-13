#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import h5py
import numpy as np


#------------------------------------------------------------------------------
#   Main execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
	train_data = './train.h5'
	valid_data = './val.h5'

	h5f_train = h5py.File(train_data, 'r')
	X_train_images = h5f_train['X']
	Y_train_labels = h5f_train['Y']

	h5f_valid = h5py.File(valid_data, 'r')
	X_valid_images = h5f_valid['X']
	Y_valid_labels = h5f_valid['Y']

	X_train_images = X_train_images[()]
	Y_train_labels = Y_train_labels[()]
	X_valid_images = X_valid_images[()]
	Y_valid_labels = Y_valid_labels[()]
	X_images = np.concatenate([X_train_images, X_valid_images], axis=0)
	Y_labels = np.concatenate([Y_train_labels, Y_valid_labels], axis=0)

	h5f = h5py.File('train_val.h5', 'w')
	h5f.create_dataset('X', data=X_images)
	h5f.create_dataset('Y', data=Y_labels)

	h5f_train.close()
	h5f_valid.close()
	h5f.close()