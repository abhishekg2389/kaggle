# kaggle
  ## Basic libraries
  * sudo apt-get update
  * sudo apt-get install python-pip python-dev build-essential build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev python-matplotlib lynx htop
  
  ## Scikit-learn
  * sudo pip install cython
  * sudo git clone https://github.com/scikit-learn/scikit-learn.git
  * cd scikit-learn
  * sudo python setup.py build
  * sudo python setup.py install
  * cd ..
  
  ## Jupyter
  * sudo pip install --upgrade pip
  * sudo pip install jupyter
  * sudo pip install jupyter_contrib_nbextensions
  * sudo jupyter contrib nbextension install
  * sudo pip install yapf
  
  ## XgBoost
  * git clone --recursive https://github.com/dmlc/xgboost
  * cd xgboost; make -j4
  * cd python-package; sudo python setup.py install
  
  ## pandas
  * pip install pandas
  * sudo apt-get install python-tables

----------------------------
sudo apt-get install -y tightvncserver

sudo apt-get install -y xfce4 xfce4-goodies

sudo apt-get install autocutsel

autocutsel -s PRIMARY -fork

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

--------------------------------------
sudo apt-get install libboost-program-options-dev libboost-python-dev libtool

git clone git://github.com/JohnLangford/vowpal_wabbit.git

cd vowpal_wabbit

./autogen.sh

./configure

make
