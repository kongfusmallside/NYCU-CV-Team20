from utils.agent import *
from utils.dataset import *



from IPython.display import clear_output

import sys
import traceback
import sys
import os
import tqdm.notebook as tq
import seaborn as sns
improve_model = "True"
improve_reward = "True"
batch_size = 32
PATH="./datasets/"

print('4')
train_loader2012, val_loader2012 = read_voc_dataset(download=LOAD, year='2012')
train_loader2007, val_loader2007 = read_voc_dataset(download=LOAD, year='2007')

classes = [ 'cat', 'dog', 'bird', 'motorbike','diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'aeroplane', 'cow', 'sheep', 'sofa']

print('5')
agents_per_class = {}
datasets_per_class = sort_class_extract([train_loader2007, train_loader2012])
#datasets_eval_per_class = sort_class_extract([val_loader2007, val_loader2012])

print("123")

"""for i in tq.tqdm(range(len(classes))):
    classe = classes[i]
    print("Classe "+str(classe)+"...")
    agents_per_class[classe] = Agent(classe, alpha=0.2, num_episodes=15, load=False)
    agents_per_class[classe].train(datasets_per_class[classe])
    del agents_per_class[classe]
    torch.cuda.empty_cache()"""
classe = classes[16]
print("Classe "+str(classe)+"...")
num_photos = len(datasets_per_class[classe])
print(f"类别 {classe} 中有 {num_photos} 张照片")
agents_per_class[classe] = Agent(classe, alpha=0.2, num_episodes=50, load=False, improve_model=improve_model, improve_reward=improve_reward)
agents_per_class[classe].train(datasets_per_class[classe])
del agents_per_class[classe]
torch.cuda.empty_cache()