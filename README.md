# kaggle


sudo apt-get install python-pip python-dev build-essential build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base python-matplotlib

sudo apt-get install nose cython git
sudo git clone https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
sudo python setup.py build
sudo python setup.py install
cd ..
nosetests -v sklearn


sudo apt-get install python-pip
sudo pip install cython
sudo git clone https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
sudo python setup.py build
sudo python setup.py install
cd ..
nosetests -v sklearn
