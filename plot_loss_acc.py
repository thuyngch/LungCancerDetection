import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Files
loss_train_csv = "ckpt/attention0.0_softmax_bs8_ep50_trainvaltest/nodule3-classifier/loss_train.csv"
loss_valid_csv = "ckpt/attention0.0_softmax_bs8_ep50_trainvaltest/nodule3-classifier/loss_val.csv"
acc_train_csv = "ckpt/attention0.0_softmax_bs8_ep50_trainvaltest/nodule3-classifier/acc_train.csv"
acc_valid_csv = "ckpt/attention0.0_softmax_bs8_ep50_trainvaltest/nodule3-classifier/acc_val.csv"

# Get reference
loss_train_df = pd.read_csv(loss_train_csv)
loss_train_steps = np.array(loss_train_df.iloc[:, 1].tolist())
loss_train_values = np.array(loss_train_df.iloc[:, 2].tolist())

loss_valid_df = pd.read_csv(loss_valid_csv)
loss_valid_steps = np.array(loss_valid_df.iloc[:, 1].tolist())
loss_valid_values = np.array(loss_valid_df.iloc[:, 2].tolist())

acc_train_df = pd.read_csv(acc_train_csv)
acc_train_steps = np.array(acc_train_df.iloc[:, 1].tolist())
acc_train_values = np.array(acc_train_df.iloc[:, 2].tolist())

acc_valid_df = pd.read_csv(acc_valid_csv)
acc_valid_steps = np.array(acc_valid_df.iloc[:, 1].tolist())
acc_valid_values = np.array(acc_valid_df.iloc[:, 2].tolist())

# Noise
mean = 0.0
std = 0.0

loss_train_values = loss_train_values + std * \
    np.random.randn(len(loss_train_values)) + mean
loss_valid_values = loss_valid_values + std * \
    np.random.randn(len(loss_valid_values)) + mean
acc_train_values = acc_train_values - std * \
    np.random.randn(len(acc_train_values)) - mean
acc_valid_values = acc_valid_values - std * \
    np.random.randn(len(acc_valid_values)) - mean

# Plot
plt.figure(dpi=100)
plt.plot(loss_train_steps, loss_train_values, '--g')
plt.plot(loss_valid_steps, loss_valid_values, '-r', lw=3)
plt.legend(['train', 'valid'])
plt.xlabel('step')
plt.ylabel('loss')
out_file = os.path.join(os.path.dirname(loss_train_csv), "loss1.png")
plt.savefig(out_file)
print("Loss is saved at {}".format(out_file))
plt.show()

plt.figure(dpi=100)
plt.plot(acc_train_steps, acc_train_values, '--g')
plt.plot(acc_valid_steps, acc_valid_values, '-r', lw=3)
plt.legend(['train', 'valid'])
plt.xlabel('step')
plt.ylabel('accuracy')
out_file = os.path.join(os.path.dirname(acc_train_csv), "acc1.png")
plt.savefig(out_file)
print("Accuracy is saved at {}".format(out_file))
plt.show()
