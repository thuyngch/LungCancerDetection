#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tflearn, h5py, cv2, itertools
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from src.models.cnn_model import CNNModel


#------------------------------------------------------------------------------
#   Functions
#------------------------------------------------------------------------------
def get_metrics(Y_test_labels, label_predictions):
    cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    precision = TP*1.0/(TP+FP)
    recall = TP*1.0/(TP+FN)
    specificity = TN*1.0/(TN+FP)
    return precision, recall, specificity, cm


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Purples):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#------------------------------------------------------------------------------
#   Get clusters
#------------------------------------------------------------------------------
# Dataset
h5f = h5py.File('src/data/train.h5', 'r')
X_train_images = h5f['X']
Y_train_labels = h5f['Y']

# Model
convnet  = CNNModel()
network = convnet.define_network(
	X_train_images, Y_train_labels, num_outputs=4,
	optimizer='adam', lr=1e-3,
	use_triplet=True, triplet_hard_mining=True,
)
model = tflearn.DNN(
	network,
	max_checkpoints=10,
	tensorboard_verbose=1,
	checkpoint_path='ckpt/nodule3-classifier.tfl.ckpt',
	best_checkpoint_path='ckpt/model_best.tfl.ckpt',
)
model.load("ckpt/model_best.tfl.ckpt8273")
# model.load("ckpt/nodule3-classifier.tfl")

# Inference
preds = model.predict(X_train_images[:,:,:,:])

# Get clusters
pos_indicator = Y_train_labels[:,1]==1
neg_indicator = Y_train_labels[:,0]==1

pos_embeddings = preds[pos_indicator,:]
neg_embeddings = preds[neg_indicator,:]

avg_pos_embedding = pos_embeddings.mean(axis=0)
avg_neg_embedding = neg_embeddings.mean(axis=0)

print("avg_pos_embedding", avg_pos_embedding)
print("avg_neg_embedding", avg_neg_embedding)


#------------------------------------------------------------------------------
#   Testing
#------------------------------------------------------------------------------
# Dataset
h5f2 = h5py.File('src/data/test.h5', 'r')
X_test_images = h5f2['X']
Y_test_labels = h5f2['Y']

# Inference
embeddings = model.predict(X_test_images[:,:,:,:])

# Cluster
pos_dists = ((embeddings-avg_pos_embedding)**2).sum(axis=1)[:,np.newaxis]
neg_dists = ((embeddings-avg_neg_embedding)**2).sum(axis=1)[:,np.newaxis]
dists = np.hstack([neg_dists, pos_dists])
pred_indicies = np.argmin(dists, axis=1)
neg_pred_indicies = np.argmax(dists, axis=1)
print("neg_ratio:", (pred_indicies==0).sum() / pred_indicies.size)
print("pos_ratio:", (pred_indicies==1).sum() / pred_indicies.size)

# Metrics
predictions = np.hstack([neg_pred_indicies[:,np.newaxis], pred_indicies[:,np.newaxis]])
precision, recall, specificity, cm = get_metrics(Y_test_labels, predictions)
print("precision:", precision)
print("recall:", recall)
print("specificity:", specificity)
