# kaggle

sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base python-pip

pip install -U scikit-learn

-------------------------------------------

sudo apt-get install python-pip python-dev build-essential build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base python-matplotlib

sudo apt-get install nose cython git

sudo git clone https://github.com/scikit-learn/scikit-learn.git

cd scikit-learn

sudo python setup.py build

sudo python setup.py install

cd ..

nosetests -v sklearn

-------------------------------------------

sudo apt-get install python-pip

sudo pip install cython

sudo git clone https://github.com/scikit-learn/scikit-learn.git

cd scikit-learn

sudo python setup.py build

sudo python setup.py install

cd ..

nosetests -v sklearn


git clone --recursive https://github.com/dmlc/xgboost

cd xgboost; make -j4

cd python-package; sudo python setup.py install
