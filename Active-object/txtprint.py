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
import tqdm.notebook as tq
import seaborn as sns
import pickle
import os
#!pip3 install torch==1.5.1 torchvision==0.6.1 -f https://download.pytorch.org/whl/cu92/torch_stable.html

batch_size = 32
PATH="./datasets/"



#train_loader2012, val_loader2012 = read_voc_dataset(download=False, year='2012')
train_loader2007, val_loader2007 = read_voc_dataset(download=LOAD, year='2007')


classes = ['dog','cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']

agents_per_class = {}
datasets_per_class = sort_class_extract([val_loader2007])
classe = classes[17]
num_photos = len(datasets_per_class[classe])
print(f"类别 {classe} 中有 {num_photos} 张照片")

torch.cuda.empty_cache()
results = {}
"""for i in classes:
    results[i] = []

for i in tq.tqdm(range(len(classes))):
    classe = classes[i]
    print("Class "+str(classe)+"...")
    agent = Agent(classe, alpha=0.2, num_episodes=50, load=True)
    res = agent.evaluate(datasets_per_class[classe])
    results[classe] = res"""

results[17] = []
classe = classes[17]
print("Class "+str(classe)+"...")
agent = Agent(classe, alpha=0.2, num_episodes=50, load=True, improve_model=False)
res = agent.evaluate_iou(datasets_per_class[classe])
results[classe] = res