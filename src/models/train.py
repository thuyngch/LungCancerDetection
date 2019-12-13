#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import numpy as np
import tflearn, h5py, os, argparse
from src.models.cnn_model import CNNModel


#------------------------------------------------------------------------------
#  Arguments
#------------------------------------------------------------------------------
# Argument parser
parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('--train_data', type=str, default='src/data/train.h5', help='Training data h5-file')

parser.add_argument('--valid_data', type=str, default='src/data/val.h5', help='Validating data h5-file')

parser.add_argument('--ckpt', type=str, default='ckpt/attention0.5_softmax_bs8/nodule3-classifier.ckpt', help='Checkpoint path')

parser.add_argument('--num_outputs', type=int, default=2, help='Number of outputs')

parser.add_argument('--hidden_embedding', type=int, default=512, help='Hidden embedding size')

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')

parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

parser.add_argument('--attention_ratio', type=float, default=0.0, help='Attention ratio')

parser.add_argument('--use_pooling', action='store_true', default=False, help='Use max pooling')

parser.add_argument('--use_bn', action='store_true', default=False, help='Use batchnorm')

parser.add_argument('--use_triplet', action='store_true', default=False, help='Use triplet loss instead of CE loss')

parser.add_argument('--triplet_hard_mining', action='store_true', default=False, help='Batch-hard triplet loss')

parser.add_argument('--resume', type=str, default=None, help='Resume ckpt path')

args = parser.parse_args()

# Take arguments
train_data = args.train_data
valid_data = args.valid_data
ckpt = args.ckpt
resume = args.resume
os.makedirs(os.path.dirname(ckpt), exist_ok=True)

use_bn = args.use_bn
num_outputs = args.num_outputs
hidden_embedding = args.hidden_embedding

lr = args.lr
epochs = args.epoch
batch_size = args.batch_size
attention_ratio = args.attention_ratio
use_pooling = args.use_pooling
use_triplet = args.use_triplet
triplet_hard_mining = args.triplet_hard_mining


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
		X_train_images, Y_train_labels,
		num_outputs=num_outputs, hidden_embedding=hidden_embedding, optimizer='adam', lr=lr,
		use_pooling=use_pooling, use_bn=use_bn, attention_ratio=attention_ratio,
		use_triplet=use_triplet, triplet_hard_mining=triplet_hard_mining,
	)
	model = tflearn.DNN(network, best_checkpoint_path=ckpt, max_checkpoints=1)

	# Resume
	if resume is not None:
		model.load(resume, weights_only=True)
		print("Load checkpoint from %s" % (resume))

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
	h5f.close()
	h5f2.close()
