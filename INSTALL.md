# Installation

* Create environment:
```bash
conda create -n lungcancer python=3.6 -y
conda activate lungcancer
```

* Clone Git repository:
```bash
git clone https://github.com/thuyngch/LungCancerDetection.git
cd LungCancerDetection
```

* Install required packages:
```bash
conda install cython numpy ipython jupyter jupyterlab -y
pip install opencv-python tqdm scikit-image albumentations pandas pylint sklearn
conda install cudatoolkit==9.0 cudnn==7.1.2 tensorflow-gpu==1.12 keras==2.2.4 -y
pip install h5py gdown tflearn==0.3.2
ipython kernel install --user --name=lungcancer
python setup.py develop
```

* Check tensorflow GPU:
```bash
ipython
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

* Download ckpt:
```bash
gdown https://drive.google.com/uc?id=1WcazKLojho9lbz-8jYPAoibH8TozfBNs
unzip ckpt.zip
rm ckpt.zip
```
