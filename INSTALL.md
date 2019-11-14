# Installation

* Create pip environment:
```
cd ~/.virtualenvs
virtualenv -p python3.6 lungcancer
workon lungcancer
```

* Clone Git repository:
```
git clone https://github.com/thuyngch/LungCancerDetection.git
cd LungCancerDetection
```

* Install required packages:
```
pip install numpy cython opencv-python tqdm scikit-image albumentations pandas pylint
pip install ipython jupyter jupyterlab
pip install tensorflow==1.13.1 tensorboard==1.13 tflearn tensorflow-gpu==1.13.1
ipython kernel install --user --name=lungcancer
python setup.py develop
```

* Config the VSCode remember the last commit message:
```
echo "update" > .mycommitmsg.txt
git config --local commit.template .mycommitmsg.txt
printf "`git log -1 --pretty=%s`" > .gitmessage.txt
chmod +x .git/hooks/post-commit
```
