#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
from cnn_model import CNNModel

import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle, argparse, h5py, os, itertools, tflearn
from sklearn.metrics import roc_curve, auc, confusion_matrix


#------------------------------------------------------------------------------
#  Utilizations
#------------------------------------------------------------------------------
def create_mosaic(image, nrows, ncols):
	"""
	Tiles all the layers in nrows x ncols
	Args:
	------
	image = 3d numpy array of M * N * number of filters dimensions
	nrows = integer representing number of images in a row
	ncol = integer representing number of images in a column

	returns formatted image
	"""

	M = image.shape[1]
	N = image.shape[2]

	npad = ((0,0), (1,1), (1,1))
	image = np.pad(image, pad_width = npad, mode = 'constant',\
		constant_values = 0)
	M += 2
	N += 2
	image = image.reshape(nrows, ncols, M, N)
	image = np.transpose(image, (0,2,1,3))
	image = image.reshape(M*nrows, N*ncols)
	return image

def format_image(image, num_images):
	"""
	Formats images
	"""
	idxs = np.random.choice(image.shape[0], num_images)
	M = image.shape[1]
	N = image.shape[2]
	imagex = np.squeeze(image[idxs, :, :, :])
	return imagex

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Purples):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	#plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	#plt.grid('off')
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def load_images(filename):
	"""
	Loads images contained in hdfs file
	"""
	h5f2 = h5py.File(filename, 'r')
	X_test_images = h5f2['X']
	Y_test_labels = h5f2['Y']
	return X_test_images, Y_test_labels

def plot_predictions(images, filename):
	"""
	Plots the predictions mosaic
	"""
	imagex = format_image(images, 4)
	mosaic = create_mosaic(imagex, 2, 2)
	plt.figure(figsize = (12, 12))
	plt.imshow(mosaic, cmap = 'gray')
	plt.axis('off')
	plt.savefig(filename + '.png', bbox_inches='tight')

def get_predictions(model, X_test_images, Y_test_labels):
	"""
	Args:
	------
	Given hdfs file of X_test_images and Y_test_labels
	
	returns:
	--------
	predictions: probability values for each class 
	label_predictions: returns predicted classes
	"""
	scores = np.vstack(model.predict(X_test_images[:,:,:,:]))
	label_predictions = np.zeros_like(scores)
	label_predictions[np.arange(len(scores)), scores.argmax(1)] = 1
	return scores, label_predictions

def get_roc_curve(Y_test_labels, predictions):
	"""
	Args:
	-------
	hdfs datasets: Y_test_labels and predictions
	
	Returns:
	--------
	fpr: false positive Rate
	tpr: true posiive Rate
	roc_auc: area under the curve value
	"""
	fpr, tpr, thresholds = roc_curve(Y_test_labels[:,1], predictions[:,1], pos_label=1)
	roc_auc = auc(fpr, tpr)
	return fpr, tpr, roc_auc

def get_metrics(Y_test_labels, label_predictions):
	"""
	Args:
	-----
	Y_test_labels, label_predictions

	Returns:
	--------
	precision, recall and specificity values and cm
	"""
	cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])

	TN = cm[0][0]
	FP = cm[0][1]
	FN = cm[1][0]
	TP = cm[1][1]

	precision = TP*1.0/(TP+FP)
	recall = TP*1.0/(TP+FN)
	specificity = TN*1.0/(TN+FP)

	return precision, recall, specificity, cm

def plot_roc_curve(fpr, tpr, roc_auc):
	"""
	Plots ROC curve

	Args:
	-----
	FPR, TPR and AUC
	"""
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.axis('equal')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.savefig('roc1.png', bbox_inches='tight')


#------------------------------------------------------------------------------
#  Evaluation
#------------------------------------------------------------------------------
def eval_softmax(model, X_test_images, Y_test_labels, tta=False):
	X_test_images = X_test_images.value

	if not tta:
		scores, label_predictions = get_predictions(model, X_test_images, Y_test_labels)

	else:
		# Prepare augmented samples
		list_X_test_images = [
			X_test_images.copy(),
			# X_test_images[:,:,::-1,:].copy(),
			X_test_images[:,:,:,::-1].copy(),
			# X_test_images[:,:,::-1,::-1].copy(),
		]
		# Get predicted scores
		scores = []
		for test_images in list_X_test_images:
			score, _ = get_predictions(model, test_images, Y_test_labels)
			scores.append(score)
		scores = np.stack(scores)
		scores = scores.mean(axis=0)

		# Get predicted labels
		label_predictions = np.zeros_like(scores)
		label_predictions[np.arange(len(scores)), scores.argmax(1)] = 1

	precision, recall, specificity, cm = get_metrics(Y_test_labels, label_predictions)
	return precision, recall, specificity


def eval_triplet(model, X_train_images, Y_train_labels, X_test_images, Y_test_labels):
	# Get clusters
	preds = model.predict(X_train_images[:,:,:,:])
	pos_indicator = Y_train_labels[:,1]==1
	neg_indicator = Y_train_labels[:,0]==1
	pos_embeddings = preds[pos_indicator,:]
	neg_embeddings = preds[neg_indicator,:]
	avg_pos_embedding = pos_embeddings.mean(axis=0)
	avg_neg_embedding = neg_embeddings.mean(axis=0)

	# Prediction
	embeddings = model.predict(X_test_images[:,:,:,:])
	pos_dists = ((embeddings-avg_pos_embedding)**2).sum(axis=1)[:,np.newaxis]
	neg_dists = ((embeddings-avg_neg_embedding)**2).sum(axis=1)[:,np.newaxis]
	dists = np.hstack([neg_dists, pos_dists])
	pred_indicies = np.argmin(dists, axis=1)
	neg_pred_indicies = np.argmax(dists, axis=1)
	predictions = np.hstack([neg_pred_indicies[:,np.newaxis], pred_indicies[:,np.newaxis]])
	precision, recall, specificity, cm = get_metrics(Y_test_labels, predictions)

	return precision, recall, specificity


#------------------------------------------------------------------------------
#  Argument parser
#------------------------------------------------------------------------------
# Argument parser
parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('--train_data', type=str, default='src/data/train.h5', help='Training data h5-file')

parser.add_argument('--test_data', type=str, default='src/data/test.h5', help='Testing data h5-file')

parser.add_argument('--ckpt', type=str, default='ckpt/attention0.5_softmax_bs8/nodule3-classifier.ckpt', help='Checkpoint path')

parser.add_argument('--num_outputs', type=int, default=2, help='Number of outputs')

parser.add_argument('--hidden_embedding', type=int, default=512, help='Hidden embedding size')

parser.add_argument('--attention_ratio', type=float, default=0.0, help='Attention ratio')

parser.add_argument('--use_pooling', action='store_true', default=False, help='Use max pooling')

parser.add_argument('--use_bn', action='store_true', default=False, help='Use batchnorm')

parser.add_argument('--use_triplet', action='store_true', default=False, help='Use triplet loss instead of CE loss')

parser.add_argument('--triplet_hard_mining', action='store_true', default=False, help='Batch-hard triplet loss')

parser.add_argument('--tta', action='store_true', default=False, help='Test time augmentation')

args = parser.parse_args()

# Take arguments
train_data = args.train_data
test_data = args.test_data
ckpt = args.ckpt
os.makedirs(os.path.dirname(ckpt), exist_ok=True)

use_bn = args.use_bn
num_outputs = args.num_outputs
hidden_embedding = args.hidden_embedding

attention_ratio = args.attention_ratio
use_pooling = args.use_pooling
use_triplet = args.use_triplet
triplet_hard_mining = args.triplet_hard_mining
tta = args.tta


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
	# Load data
	X_train_images, Y_train_labels = load_images(train_data)
	X_test_images, Y_test_labels = load_images(test_data)

	# Build model
	convnet  = CNNModel()
	network = convnet.define_network(
		X_test_images, Y_test_labels, num_outputs=num_outputs, hidden_embedding=hidden_embedding,
		use_pooling=use_pooling, use_bn=use_bn, attention_ratio=attention_ratio,
		use_triplet=use_triplet, triplet_hard_mining=triplet_hard_mining,
	)
	model = tflearn.DNN(network)

	ckpt = tf.train.latest_checkpoint(ckpt)
	print("ckpt", ckpt)
	model.load(ckpt)

	# Model prediction
	if not use_triplet:
		precision, recall, specificity = eval_softmax(model, X_test_images, Y_test_labels, tta)
	else:
		precision, recall, specificity = eval_triplet(model, X_train_images, Y_train_labels, X_test_images, Y_test_labels, tta)

	print("precision: %.6f" % (precision))
	print("recall: %.6f" % (recall))
	print("specificity: %.6f" % (specificity))
