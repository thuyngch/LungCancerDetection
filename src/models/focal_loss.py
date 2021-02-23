import tensorflow as tf

def focal_loss_sigmoid(y_pred, labels, alpha=0.25, gamma=2.0):
	labels = labels[:,1]
	L = -labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred) - (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
	L = tf.reduce_mean(L)
	return L

def focal_loss_softmax(y_pred, labels, gamma=2):
	L = -labels * ((1.0-y_pred)**gamma) * tf.log(y_pred)
	L = tf.reduce_mean(L)
	return L
