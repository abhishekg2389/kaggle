from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# kaggle
  ## Basic libraries
  * sudo apt-get update
  * sudo apt-get install python-pip python-dev build-essential build-essential python-dev python-setuptools python-numpy python-scipy libatlas-base-dev python-matplotlib lynx htop git
  * sudo pip install scikit-learn future jupyter jupyter_contrib_nbextensions yapf pandas seaborn hyperopt lightgbm
  
  ## Scikit-learn
  * sudo pip install scikit-learn
  
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
  
  1. `openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mykey.key -out mycert.pem`
  2. `jupyter notebook --generate-config`
  3. `jupyter notebook password`
  4. `nano ~/.jupyter/jupyter_notebook_config.json` with below configurations
```
{
  "NotebookApp": {
    "password": "sha1:bbd262748b3b:e28e5c7846dc72a7a07b67096e1938711dca37d7",
    "certfile": "/home/ubuntu/mycert.pem",
    "keyfile": "/home/ubuntu/mykey.key",
    "ip": "*",
    "port": 9999
  }
}
```
  5. `nohup jupyter notebook &`
  
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
