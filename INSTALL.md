# Installation

* Create environment:
```
conda create -n lungcancer python=3.6 -y
conda activate lungcancer
```

* Clone Git repository:
```
thuync
git clone https://github.com/thuyngch/LungCancerDetection.git
cd LungCancerDetection
```

* Install required packages:
```
conda install cython numpy ipython jupyter jupyterlab -y
pip install opencv-python tqdm scikit-image albumentations pandas pylint sklearn
conda install cudatoolkit==9.0 cudnn==7.1.2 tensorflow-gpu==1.12 keras==2.2.4 -y
pip install tflearn h5py
ipython kernel install --user --name=lungcancer
python setup.py develop
```

* Check tensorflow GPU:
```
ipython
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```
