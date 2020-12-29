"""
Builds a HDF5 data set for test, train and validation data
Run script as python build_hdf5_datasets.py $mode
where mode can be 'test', 'train', 'val'
"""
from tflearn.data_utils import build_hdf5_image_dataset
from glob import glob
import random
import h5py
import sys
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Check inputs
if len(sys.argv) < 2:
    raise ValueError(
        '1 argument needed. Specify if you need to generate a train, test or val set')
else:
    mode = sys.argv[1]
    if mode not in ['train', 'test', 'val']:
        raise ValueError(
            'Argument not recognized. Has to be train, test or val')

pos_img_dir = "/home/thuync/Workspace/Luna16_git/input/extract/pos"
neg_img_dir = "/home/thuync/Workspace/Luna16_git/input/extract/neg"

dataset_file = os.path.join("src/data", '{}_datalabels.txt'.format(mode))
h5_dataset_file = os.path.join("src/data", '{}_dataset.h5'.format(mode))
h5_file = os.path.join("src/data", '{}_s100.h5'.format(mode))


# Get image files
pos_img_files = sorted(glob(os.path.join(pos_img_dir, "*.jpg")))
neg_img_files = sorted(glob(os.path.join(neg_img_dir, "*.jpg")))
print("pos_img_files:", len(pos_img_files))
print("neg_img_files:", len(neg_img_files))

random.seed(0)
random.shuffle(pos_img_files)
random.seed(0)
random.shuffle(neg_img_files)

train_pos_img_files = pos_img_files[:1200]
val_pos_img_files = pos_img_files[1200:1200+400]
test_pos_img_files = pos_img_files[1200+400:1200+400+400]

train_neg_img_files = neg_img_files[:4800]
val_neg_img_files = neg_img_files[4800:4800+1600]
test_neg_img_files = neg_img_files[4800+1600:4800+1600+1600]

if mode == 'train':
    filenames = [*train_pos_img_files, *train_neg_img_files]
    labels = [1] * len(train_pos_img_files) + [0] * len(train_neg_img_files)
elif mode == 'val':
    filenames = [*val_pos_img_files, *val_neg_img_files]
    labels = [1] * len(val_pos_img_files) + [0] * len(val_neg_img_files)
else:
    filenames = [*test_pos_img_files, *test_neg_img_files]
    labels = [1] * len(test_pos_img_files) + [0] * len(test_neg_img_files)

with open(dataset_file, 'w') as fp:
    for (filename, label) in zip(filenames, labels):
        line = "{} {}".format(filename, label)
        fp.writelines(line+'\n')

build_hdf5_image_dataset(
    dataset_file, image_shape=(100, 100, 1),
    mode='file', output_path=h5_dataset_file,
    categorical_labels=True, normalize=True, grayscale=True)

# Load HDF5 dataset
h5f = h5py.File(h5_dataset_file, 'r')
X_images = h5f['X']
Y_labels = h5f['Y'][:]

print(X_images.shape)
X_images = X_images[:, :, :].reshape([-1, 100, 100, 1])
print(X_images.shape)
h5f.close()

h5f = h5py.File(h5_file, 'w')
h5f.create_dataset('X', data=X_images)
h5f.create_dataset('Y', data=Y_labels)
h5f.close()
