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

pip install pandas

sudo apt-get install python-tables

----------------------------
sudo apt-get install -y tightvncserver

sudo apt-get install -y xfce4 xfce4-goodies

sudo apt-get install autocutsel

Add line in ~/.vnc/tightvncconfig or use in terminal in VNC Viewer : autocutsel -fork

------------------------------------
Matlab on server : http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server

import matplotlib as mpl;mpl.use('Agg');import matplotlib.pyplot as plt

------------------------------------
Dropbox-Uploader : git clone https://github.com/andreafabrizi/Dropbox-Uploader/

cd Dropbox-Uploader/

chmod +x dropbox_uploader.sh

./dropbox_uploader.sh

Provide full access while creating and configuring app and then use following command to upload

Dropbox-Uploader/dropbox_uploader.sh upload * /*

-------------------------------------
sudo apt-get -y install r-base

http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libmf.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libmf+zip
