# Installation

* Requirements:
```
python>=3.6
CUDA==9.0
```

* Create pip environment:
```
cd ~/.virtualenvs
virtualenv -p python3.6 lungcancer
workon lungcancer
```

* Clone Git repository:
```
thuync
git clone https://github.com/thuyngch/LungCancerDetection.git
cd LungCancerDetection
```

* Install required packages:
```
pip install numpy cython opencv-python tqdm scikit-image albumentations pandas pylint sklearn
pip install ipython jupyter jupyterlab
pip install tensorflow==1.12 tensorboard==1.12 tflearn tensorflow-gpu==1.12
ipython kernel install --user --name=lungcancer
python setup.py develop
```

* Check tensorflow GPU:
```
ipython
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

* Config the VSCode remember the last commit message:
```
echo "update" > .mycommitmsg.txt
git config --local commit.template .mycommitmsg.txt
printf "`git log -1 --pretty=%s`" > .gitmessage.txt
chmod +x .git/hooks/post-commit
```
