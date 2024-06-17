import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from image_helper import *
from parse_xml_annotations import *
from features import *
from reinforcement import *
from metrics import *
from collections import namedtuple
import time
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import imageio.v2 as imageio

print(torch.cuda.is_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def draw_bouding_box_1(annotation, annotation_1,img, x):
    #new_img = Image.fromarray(img)
    #draw = ImageDraw.Draw(new_img)
    length = len(annotation)
    annotation = np.array(annotation)
    length_1 = len(annotation_1)
    annotation_1 = np.array(annotation_1)
    new_img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(new_img)
    for i in range(length):
        
        if i==0:
            x_min = int(annotation[i,1])
            x_max = int(annotation[i,2])
            y_min = int(annotation[i,3])
            y_max = int(annotation[i,4])
            
            draw.line(((x_min, y_min), (x_max, y_min)), fill="green", width=3)
            draw.line(((x_min, y_min), (x_min, y_max)), fill="green", width=3)
            draw.line(((x_max, y_min), (x_max, y_max)), fill="green", width=3)
            draw.line(((x_min, y_max), (x_max, y_max)), fill="green", width=3)
        
        
        
        

        
    for i in range(length_1):
        #new_img = Image.fromarray(img.copy())
        #draw = ImageDraw.Draw(new_img)
       if i==length-1:
            x_min = int(annotation_1[i,1])
            x_max = int(annotation_1[i,2])
            y_min = int(annotation_1[i,3])
            y_max = int(annotation_1[i,4])
            
            draw.line(((x_min, y_min), (x_max, y_min)), fill="red", width=3)
            draw.line(((x_min, y_min), (x_min, y_max)), fill="red", width=3)
            draw.line(((x_max, y_min), (x_max, y_max)), fill="red", width=3)
            draw.line(((x_min, y_max), (x_max, y_max)), fill="red", width=3)
       
    plt.figure()
    plt.imshow(new_img)
    plt.show()
    plt.savefig(str(i)+'.png', dpi=100)
    
    tested = 0
    while os.path.isfile('media/movie_'+str(tested)+'.gif'):
        tested += 1
    # filepaths
    fp_out = "media/movie_"+str(tested)+".gif"
    images = []
    for count in range(length):
        images.append(imageio.imread(str(count)+".png"))
    
    imageio.mimsave(fp_out, images)
    
    #for count in range(length):
        #os.remove(str(count)+".png")
    #plt.figure()
    #plt.imshow(new_img)
    #plt.show()
    return img

def get_annotation(offset, size_mask):
    annotation = np.zeros(5)
    annotation[3] = offset[0]
    annotation[4] = offset[0] + size_mask[0]
    annotation[1] = offset[1]
    annotation[2] = offset[1] + size_mask[1] 
    return annotation

print("load images")
path_voc = "./datas/VOCdevkit/VOC2007test"
image_names = np.array(load_images_names_in_data_set('aeroplane_test', path_voc))
labels = load_images_labels_in_data_set('aeroplane_test', path_voc)
image_names_aero = []
for i in range(len(image_names)):
    if labels[i] == '1':
        image_names_aero.append(image_names[i])
image_names = image_names_aero
images = get_all_images(image_names, path_voc)
print("aeroplane_test image:%d" % len(image_names))



#Q_NETWORK_PATH = './models/' + 'voc2012_2007_model'
#model = get_q_network(weights_path=Q_NETWORK_PATH).cuda()
Q_NETWORK_PATH = './models/' + 'voc2012_2007_model'
model = torch.load(Q_NETWORK_PATH)
model_vgg = getVGG_16bn("./models")
model_vgg = model_vgg.cuda()

class_object = 1
steps = 10
res = []
res_step = []
res_annotations = []
re_IOU = []
for i in range(len(image_names)):
        image_name = image_names[i]
        image = images[i]
        
        # get use for iou calculation
        gt_annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
        original_shape = (image.shape[0], image.shape[1])
        classes_gt_objects = get_ids_objects_from_annotation(gt_annotation)
        gt_masks = generate_bounding_box_from_annotation(gt_annotation, image.shape)
        
        # the initial part
        region_image = image
        size_mask = image.shape
        region_mask = np.ones((image.shape[0], image.shape[1]))
        offset = (0, 0)
        history_vector = torch.zeros((4,6))
        state = get_state(region_image, history_vector, model_vgg)
        done = False
        
        # save the bounding box maked by agent
        annotations = []
        annotation = get_annotation(offset, size_mask)
        annotations.append(annotation)
        
        for step in range(steps):
                # Select action
                qval = model(Variable(state))
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] + 1
                # Perform the action and observe new state
                if action == 6:
                    next_state = None
                    done = True
                else:
                    offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,
                                                                    region_image, size_mask, action)
                
                    #print(offset, size_mask, region_image.shape, region_mask.shape)
                    annotation = get_annotation(offset, size_mask)
                    annotations.append(annotation)
                    history_vector = update_history_vector(history_vector, action)
                    next_state = get_state(region_image, history_vector, model_vgg)

                # Move to the next state
                state = next_state

                iou_test = find_max_bounding_box_origin(gt_masks, region_mask, classes_gt_objects, class_object)
                if iou_test > 0.5 or step == 9:
                    done = True

                if done:
                    res_step.append(step)
                    res_annotations.append((gt_annotation, annotations, image))
                    break
                    
        iou = find_max_bounding_box_origin(gt_masks, region_mask, classes_gt_objects, class_object)
        pos = 0
        reward = qval.data[0,5]
        if iou > 0.5:
            pos = 1

        res.append((reward, pos))
        
        re_IOU.append((iou, pos))

avg_IOU = sum([item[0] for item in re_IOU])/len([item[0] for item in re_IOU])
avg_pos = sum([item[1] for item in re_IOU])/len([item[1] for item in re_IOU])
#print(re_IOU)
    
print("Average IOU:", avg_IOU)
print("Average pos:", avg_pos)

print([item[0] for item in re_IOU])
print([item[1] for item in re_IOU])
    
res_step = np.array(res_step) + 1
plt.hist(res_step)
plt.title('Number of regions analyze per object')
plt.xlabel('Number of regions')
plt.show()
print(res_step)
print(len(res_step))

begin = 77
end = begin + 3
for i in range(begin, end):
    gt_annotation, annotation, image = res_annotations[i]
    #draw_bouding_box_1(gt_annotation, image, 1)
    #draw_bouding_box_1(annotation, img, 2)
    draw_bouding_box_1(gt_annotation, annotation, image, 1)
    plt.figure()
    plt.imshow(img_1)
    plt.show()

"""all_avg_IOU = sum([item[0] for item in all])/len([item[0] for item in all])
all_avg_pos = sum([item[1] for item in all])/len([item[1] for item in all])
print(all_avg_IOU)
print(all_avg_pos)
#作圖
avg_IOU_values = [item[0] for item in all]
avg_pos_values = [item[1] for item in all]
x_values = list(range(40))
# 绘制平均 IOU 值的图形
plt.plot(x_values, avg_IOU_values, label='Average IOU')
# 绘制平均位置值的图形
plt.plot(x_values, avg_pos_values, label='Average Position')
# 添加图例
plt.legend()
# 添加标题和标签
plt.title('Average IOU and Average Position')
plt.xlabel('Epochs')
plt.ylabel('Values')
# 显示图形
plt.show()"""


"""
Q_NETWORK_PATH = './models/' + 'voc2012_2007_model_34'
model = get_q_network(weights_path=Q_NETWORK_PATH).cuda()
#Q_NETWORK_PATH = './models/' + 'voc2012_2007_model'
#model = torch.load(Q_NETWORK_PATH)
model_vgg = getVGG_16bn("./models")
model_vgg = model_vgg.cuda()

class_object = 1
steps = 10
res = []
res_step = []
res_annotations = []
for i in range(len(image_names)):
    image_name = image_names[i]
    image = images[i]
    
    # get use for iou calculation
    gt_annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
    original_shape = (image.shape[0], image.shape[1])
    classes_gt_objects = get_ids_objects_from_annotation(gt_annotation)
    gt_masks = generate_bounding_box_from_annotation(gt_annotation, image.shape)
    
    # the initial part
    region_image = image
    size_mask = image.shape
    region_mask = np.ones((image.shape[0], image.shape[1]))
    offset = (0, 0)
    history_vector = torch.zeros((4,6))
    state = get_state(region_image, history_vector, model_vgg)
    done = False
    
    # save the bounding box maked by agent
    annotations = []
    annotation = get_annotation(offset, size_mask)
    annotations.append(annotation)
    
    for step in range(steps):
            # Select action
            qval = model(Variable(state))
            _, predicted = torch.max(qval.data,1)
            action = predicted[0] + 1
            # Perform the action and observe new state
            if action == 6:
                next_state = None
                done = True
            else:
                offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,
                                                                   region_image, size_mask, action)
               
                #print(offset, size_mask, region_image.shape, region_mask.shape)
                annotation = get_annotation(offset, size_mask)
                annotations.append(annotation)
                history_vector = update_history_vector(history_vector, action)
                next_state = get_state(region_image, history_vector, model_vgg)

            # Move to the next state
            state = next_state
            if done:
                res_step.append(step)
                res_annotations.append((gt_annotation, annotations, image))
                break
                
    iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, class_object)
    pos = 0
    reward = qval.data[0,5]
    if iou > 0.5:
        pos = 1

    res.append((reward, pos))
    re_IOU = []
    re_IOU.append(iou)

avg_IOU = np.mean(re_IOU)
print("Average IOU:", avg_IOU)
"""

"""
begin = 10
end = begin + 10
for i in range(begin, end):
    gt_annotation, annotation, image = res_annotations[i]
    draw_bouding_box_1(gt_annotation, image)
    draw_bouding_box_1(annotation, image)
"""

"""
path_voc = "./datas/VOCdevkit/VOC2007"

# get models 
print("load models")

model_vgg = getVGG_16bn("./models")
model_vgg = model_vgg.cuda()
model = get_q_network()
model = model.cuda()

# define optimizers for each model
optimizer = optim.Adam(model.parameters(),lr=1e-6)
criterion = nn.MSELoss().cuda()   

# get image datas
path_voc_1 = "./datas/VOCdevkit/VOC2007"
path_voc_2 = "./datas/VOCdevkit/VOC2012"
class_object = '1'
image_names_1, images_1 = load_image_data(path_voc_1, class_object)
image_names_2, images_2 = load_image_data(path_voc_2, class_object)
image_names = image_names_1 + image_names_2
images = images_1 + images_2

print("aeroplane_trainval image:%d" % len(image_names))

# define the Pytorch Tensor
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# define the super parameter
epsilon = 1.0
BATCH_SIZE = 100
GAMMA = 0.90
CLASS_OBJECT = 1
steps = 10
epochs = 50
memory = ReplayMemory(1000)

def select_action(state):
    if random.random() < epsilon:
        action = np.random.randint(1,7)
    else:
        qval = model(Variable(state))
        _, predicted = torch.max(qval.data,1)
        action = predicted[0] + 1
    return action


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
def optimizer_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = Variable(torch.cat(next_states), 
                                     volatile=True).type(Tensor)
    state_batch = Variable(torch.cat(batch.state)).type(Tensor)
    action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
    reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(Tensor)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE, 1).type(Tensor)) 
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].unsqueeze(1)
    
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute  loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# train procedure
print('train the Q-network')
for epoch in range(epochs):
    print('epoch: %d' %epoch)
    now = time.time()
    for i in range(len(image_names)):
        # the image part
        image_name = image_names[i]
        image = images[i]
        if i < len(image_names_1):
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_1)
        else:
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc_2)            
        classes_gt_objects = get_ids_objects_from_annotation(annotation)
        gt_masks = generate_bounding_box_from_annotation(annotation, image.shape) 
         
        # the iou part
        original_shape = (image.shape[0], image.shape[1])
        region_mask = np.ones((image.shape[0], image.shape[1]))
        #choose the max bouding box
        iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
        
        # the initial part
        region_image = image
        size_mask = original_shape
        offset = (0, 0)
        history_vector = torch.zeros((4,6))
        state = get_state(region_image, history_vector, model_vgg)
        done = False
        for step in range(steps):
            # Select action, the author force terminal action if case actual IoU is higher than 0.5
            if iou > 0.5:
                action = 6
            else:
                action = select_action(state)
            
            # Perform the action and observe new state
            if action == 6:
                next_state = None
                reward = get_reward_trigger(iou)
                done = True
            else:
                offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,
                                                                   region_image, size_mask, action)
                # update history vector and get next state
                history_vector = update_history_vector(history_vector, action)
                next_state = get_state(region_image, history_vector, model_vgg)
                
                # find the max bounding box in the region image
                new_iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
                reward = get_reward_movement(iou, new_iou)
                iou = new_iou
            print('epoch: %d, image: %d, step: %d, reward: %d' %(epoch ,i, step, reward))    
            # Store the transition in memory
            memory.push(state, action-1, next_state, reward)
            
            # Move to the next state
            state = next_state
            
            # Perform one step of the optimization (on the target network)
            optimizer_model()
            if done:
                break
    if epsilon > 0.1:
        epsilon -= 0.1
    time_cost = time.time() - now
    print('epoch = %d, time_cost = %.4f' %(epoch, time_cost))
    Q_NETWORK_PATH_epoch = './models/' + 'voc2012_2007_model_' + str(epoch)
    torch.save(model.state_dict(), Q_NETWORK_PATH_epoch)
    print('Save')
    
# save the whole model
Q_NETWORK_PATH = './models/' + 'voc2012_2007_model'
torch.save(model, Q_NETWORK_PATH)
print('Complete')
"""