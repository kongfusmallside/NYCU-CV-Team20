from IPython.display import clear_output
from utils.agent import *
from utils.dataset import *
from utils.models import *


import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
import sys
from torch.autograd import Variable
import traceback
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

#!pip3 install torch==1.5.1 torchvision==0.6.1 -f https://download.pytorch.org/whl/cu92/torch_stable.html

batch_size = 32
PATH="./datasets/"



#train_loader2012, val_loader2012 = read_voc_dataset(download=False, year='2012')
train_loader2007, val_loader2007 = read_voc_dataset(download=LOAD, year='2007')



agents_per_class = {}
datasets_per_class = sort_class_extract([val_loader2007])
classe = 'aeroplane'
index = random.choice(list(datasets_per_class[classe].keys()))
agent = Agent(classe, alpha=0.2, num_episodes=50, load=True,  improve_model=False)

for i in range(1):
    print(list(datasets_per_class[classe].keys())[79])
    #index = random.choice(list(datasets_per_class[classe].keys()))
    image, gt_boxes = extract(list(datasets_per_class[classe].keys())[79], datasets_per_class[classe])
    agent.predict_image(image, gt_boxes, plot=True)