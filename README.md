# Pokemon-Image-Classifier
Its a convNet built upon InceptionV3 and trained on 928 pokemon classes. The model can predict any pokemon image till season 20 (Pokemon Sun and Moon)
> **Model details**:  
 loss: **0.1279** - accuracy: **0.9743** - validation loss: **0.9940** - validation accuracy: **0.7917**

## Getting Started

Clone the repository to your local machine  
Extract all the files into a directory  
Run the  	**Run_model.ipynb** 

### Prerequisites

Dependencies :  
* cv2  
* Matplotlib
* Tensorflow - gpu
* keras
* pillow
* pickle
* os
* requests
* io
```
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Model,load_model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,Input,GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from os import path
from os import listdir
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import pickle
```

### Installing


Cv2

```
$ pip install opencv-python (if you need only main modules)
$ pip install opencv-contrib-python (if you need both main and contrib modules)
```

Matplotlib

```
$ pip install matplotlib

If you are on Linux, you might prefer to use your package manager. Matplotlib is packaged for almost every major Linux distribution.

    Debian / Ubuntu: sudo apt-get install python3-matplotlib
    Fedora: sudo dnf install python3-matplotlib
    Red Hat: sudo yum install python3-matplotlib
    Arch: sudo pacman -S python-matplotlib

```
Tensorflow GPU

```
$ pip install tensorflow-gpu  # stable

$ pip install tf-nightly-gpu  # preview

You will also need to install CUDA drivers, for more details visit  
https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
```

Keras

```
These installation steps assume that you are on a Linux or Mac environment. If you are on Windows, you will need to remove sudo to run the commands below.
$ sudo pip install keras  

$ pip install keras

Alternatively: install Keras from the GitHub source:
$ git clone https://github.com/keras-team/keras.git
Then, cd to the Keras folder and run the install command:
$ cd keras
$ sudo python setup.py install
```
pillow 

```
$ pip install Pillow
$ easy_install Pillow
$ python setup.py install
```
Screen Shots :  
Screenshot 1
![Pikachu](https://github.com/AbdulAhadSiddiqui11/Pokemon-Image-Classifier/blob/master/SnapShots/poke_snap1.JPG)  

Screenshot 2
![Charmander](https://github.com/AbdulAhadSiddiqui11/Pokemon-Image-Classifier/blob/master/SnapShots/poke_snap2.JPG)  

Screenshot 3
![Darkrai](https://github.com/AbdulAhadSiddiqui11/Pokemon-Image-Classifier/blob/master/SnapShots/poke_snap3.JPG)  

Screenshot 4
![Kyurem(Black)](https://github.com/AbdulAhadSiddiqui11/Pokemon-Image-Classifier/blob/master/SnapShots/poke_snap4.JPG)  


## Deployment

### Training the model
 Detailed instructions on how to train this model are described in the jupyter notebook attached ('**PokeClassifier_Train.ipynb**').

### Using pre-trained weights
 If you want to download and use the weights of this trained model follow the link below.   
> https://drive.google.com/file/d/1Zai3RoV7L7mX1AlUqqYgUHb1VHuQ3N5A/view?usp=sharing

### Using the Model
 You can directly run the app to predict the pokemon in an image
 ####  Instructions : 
> 1.    Download the model.h5 from the link provided  
> 2(a). Download the pokemon_classes pickle file  
> 2(b). Download the dataset from the link provided (optional) to create the classes_list (skip this if you're using the pickle file)  
> 3.    Either run Run_model.ipynb or Run_model.py
> Note : Change the locations to files if needed.  

### Other details  
 Python files for all the notebooks are also provided, if you need .py scripts for some reason.  

## Contributing

Please read [CODE_OF_CONDUCT.md](https://github.com/AbdulAhadSiddiqui11/Pokemon-Image-Classifier/blob/master/CODE_OF_CONDUCT.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Abdul Ahad Siddiqui** - *Initial work* - [Abdul Ahad Siddiqui](https://github.com/AbdulAhadSiddiqui11)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/AbdulAhadSiddiqui11/Pokemon-Image-Classifier/blob/master/LICENSE) file for details
