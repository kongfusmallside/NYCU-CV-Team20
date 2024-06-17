#import torchvision.datasets.SBDataset as sbd
from utils.models import *
from utils.tools import *
import os
import imageio
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

from tqdm.notebook import tqdm
from config import *

import glob
from PIL import Image

class Agent():
    def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False, improve_model=False, improve_reward=False ):
        """
            Classe initialisant l'ensemble des paramètres de l'apprentissage, un agent est associé à une classe donnée du jeu de données.
        """
        self.BATCH_SIZE = 256
        self.GAMMA = 0.900
        self.EPS = 1
        self.TARGET_UPDATE = 1
        self.save_path = SAVE_MODEL_PATH
        screen_height, screen_width = 224, 224
        self.n_actions = 9
        self.classe = classe
        self.improve_reward = improve_reward
        self.feature_extractor = FeatureExtractor()
        if not load:
            if improve_model:
                print("improve dqn")
                self.policy_net = DuelingDQN(25169, 9)
            else:
                self.policy_net = DQN(screen_height, screen_width, self.n_actions)
        else:
            self.policy_net = self.load_network()

        if improve_model:
            print("improve dqn")    
            self.target_net = DuelingDQN(25169, 9)
        else:
            self.target_net = DQN(screen_height, screen_width, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.feature_extractor.eval()
        if use_cuda:
          self.feature_extractor = self.feature_extractor.cuda()
          self.target_net = self.target_net.cuda()
          self.policy_net = self.policy_net.cuda()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        
        self.alpha = alpha # €[0, 1]  Scaling factor
        self.nu = nu # Reward of Trigger
        self.threshold = threshold
        self.actions_history = []
        self.num_episodes = num_episodes
        self.actions_history += [[100]*9]*20

    def save_network(self, i_epoch, aa):
        """
            Fonction de sauvegarde du Q-Network
        """
        torch.save(self.policy_net, self.save_path+"_"+self.classe+"_epoch"+str(i_epoch)+"_"+str(aa))
        print('Saved')

    def load_network(self):
        """
            Récupération d'un Q-Network existant
        """
        if not use_cuda:
            return torch.load(self.save_path+"_"+self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path+"_"+self.classe+"_epoch49_1")



    def intersection_over_union(self, box1, box2):
        """
            Calcul de la mesure d'intersection/union
            Entrée :
                Coordonnées [x_min, x_max, y_min, y_max] de la boite englobante de la vérité terrain et de la prédiction
            Sortie :
                Score d'intersection/union.

        """
        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou
    
    def calculate_gtc(self, bbox, gtbbox):
        gt_area = (gtbbox[1] - gtbbox[0]) * (gtbbox[3] - gtbbox[2])
        
        intersect_width = max(min(gtbbox[1], bbox[1]) - max(gtbbox[0], bbox[0]), 0)
        intersect_height = max(min(gtbbox[3], bbox[3]) - max(gtbbox[2], bbox[2]), 0)
        intersect_area = intersect_width * intersect_height
        gtc = intersect_area / gt_area
        return gtc

    def calculate_cd(self, bbox, gtbbox):
        image_shape =(224, 224)
        diagonal_length = np.linalg.norm(np.array(image_shape))
 
        gt_center = np.array([(gtbbox[2] + gtbbox[3]) / 2, (gtbbox[0] + gtbbox[1]) / 2])
        pred_center = np.array([(bbox[2] + bbox[3]) / 2, (bbox[0] + bbox[1]) / 2])
        cd = np.linalg.norm(gt_center - pred_center)
        normalized_cd = cd / diagonal_length
        #print(normalized_cd)
        return normalized_cd
    
    def compute_reward(self, actual_state, previous_state, ground_truth):
        """
            Calcul la récompense à attribuer pour les états non-finaux selon les cas.
            Entrée :
                Etats actuels et précédents ( coordonnées de boite englobante )
                Coordonnées de la vérité terrain
            Sortie :
                Récompense attribuée
        """
        actual_reward = self.intersection_over_union(actual_state, ground_truth) + self.calculate_gtc(actual_state, ground_truth) + 1 - self.calculate_cd(actual_state, ground_truth)
        previous_reward = self.intersection_over_union(previous_state, ground_truth) + self.calculate_gtc(previous_state, ground_truth) + 1 - self.calculate_cd(previous_state, ground_truth)
        if self.improve_reward:
            #print("improve reward")
            #print(actual_reward)
            res = actual_reward - previous_reward
            #print(res)
        else:
            res = self.intersection_over_union(actual_state, ground_truth) - self.intersection_over_union(previous_state, ground_truth)
        if res <= 0:
            return -1
        return 1
      
    def rewrap(self, coord):
        return min(max(coord,0), 224)
      
    def compute_trigger_reward(self, actual_state, ground_truth):
        """
            Calcul de la récompensée associée à un état final selon les cas.
            Entrée :
                Etat actuel et boite englobante de la vérité terrain
            Sortie : 
                Récompense attribuée
        """
        res = self.intersection_over_union(actual_state, ground_truth)
        if res>=self.threshold:
            return self.nu
        return -1*self.nu

    def get_best_next_action(self, actions, ground_truth):
        """
            Implémentation de l'Agent expert qui selon l'état actuel et la vérité terrain va donner la meilleur action possible.
            Entrée :
                - Liste d'actions executées jusqu'à présent.
                - Vérité terrain.
            Sortie :
                - Indice de la meilleure action possible.

        """
        max_reward = -99
        best_action = -99
        positive_actions = []
        negative_actions = []
        actual_equivalent_coord = self.calculate_position_box(actions)
        for i in range(0, 9):
            copy_actions = actions.copy()
            copy_actions.append(i)
            new_equivalent_coord = self.calculate_position_box(copy_actions)
            if i!=0:
                reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, ground_truth)
            else:
                reward = self.compute_trigger_reward(new_equivalent_coord,  ground_truth)
            
            if reward>=0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions)==0:
            return random.choice(negative_actions)
        return random.choice(positive_actions)


    def select_action(self, state, actions, ground_truth):
        """
            Selection de l'action dépendemment de l'état
            Entrée :
                - Etat actuel. 
                - Vérité terrain.
            Sortie :
                - Soi l'action qu'aura choisi le modèle soi la meilleure action possible ( Le choix entre les deux se fait selon un jet aléatoire ).
        """
        sample = random.random()
        eps_threshold = self.EPS
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                try:
                  return action.cpu().numpy()[0]
                except:
                  return action.cpu().numpy()
        else:
            #return np.random.randint(0,9)   # Avant implémentation d'agent expert
            return self.get_best_next_action(actions, ground_truth) # Appel à l'agent expert.

    def select_action_model(self, state):
        """
            Selection d'une action par le modèle selon l'état
            Entrée :
                - Etat actuel ( feature vector / sortie du réseau convolutionnel + historique des actions )
            Sortie :
                - Action séléctionnée.
        """
        with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                #print("Predicted : "+str(qval.data))
                action = predicted[0] # + 1
                #print(action)
                return action

    def optimize_model(self):
        """
        Fonction effectuant les étapes de mise à jour du réseau ( sampling des épisodes, calcul de loss, rétro propagation )
        """
        # Si la taille actuelle de notre mémoire est inférieure aux batchs de mémoires qu'on veut prendre en compte on n'effectue
        # Pas encore d'optimization
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Extraction d'un echantillon aléatoire de la mémoire ( ou chaque éléments est constitué de (état, nouvel état, action, récompense) )
        # Et ce pour éviter le biais occurant si on apprenait sur des états successifs
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Séparation des différents éléments contenus dans les différents echantillons
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        next_states = [s for s in batch.next_state if s is not None]
        with torch.no_grad():
            non_final_next_states = Variable(torch.cat(next_states)).type(Tensor)
        
        state_batch = Variable(torch.cat(batch.state)).type(Tensor)
        if use_cuda:
            state_batch = state_batch.cuda()
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(Tensor)


        # Passage des états par le Q-Network ( en calculate Q(s_t, a) ) et on récupére les actions sélectionnées
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Calcul de V(s_{t+1}) pour les prochain états.
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE, 1).type(Tensor)) 

        if use_cuda:
            non_final_next_states = non_final_next_states.cuda()
        
        # Appel au second Q-Network ( celui de copie pour garantir la stabilité de l'apprentissage )
        with torch.no_grad():
            d = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = d.max(1)[0].view(-1, 1)

        # On calcule les valeurs de fonctions Q attendues ( en faisant appel aux récompenses attribuées )
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Calcul de la loss
        loss = criterion(state_action_values, expected_state_action_values)

        # Rétro-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def compose_state(self, image, dtype=FloatTensor):
        """
            Composition d'un état : Feature Vector + Historique des actions
            Entrée :
                - Image ( feature vector ). 
            Sortie :
                - Représentation d'état.
        """
        image_feature = self.get_features(image, dtype)
        image_feature = image_feature.view(1,-1)
        #print("image feature : "+str(image_feature.shape))
        history_flatten = self.actions_history.view(1,-1).type(dtype)
        state = torch.cat((image_feature, history_flatten), 1)
        return state
    
    def get_features(self, image, dtype=FloatTensor):
        """
            Extraction du feature vector à partir de l'image.
            Entrée :
                - Image
            Sortie :
                - Feature vector
        """
        global transform
        #image = transform(image)
        image = image.view(1,*image.shape)
        image = Variable(image).type(dtype)
        if use_cuda:
            image = image.cuda()
        feature = self.feature_extractor(image)
        #print("Feature shape : "+str(feature.shape))
        return feature.data

    
    def update_history(self, action):
        """
            Fonction qui met à jour l'historique des actions en y ajoutant la dernière effectuée
            Entrée :
                - Dernière action effectuée
        """
        action_vector = torch.zeros(9)
        action_vector[action] = 1
        size_history_vector = len(torch.nonzero(self.actions_history))
        if size_history_vector < 9:
            self.actions_history[size_history_vector][action] = 1
        else:
            for i in range(8,0,-1):
                self.actions_history[i][:] = self.actions_history[i-1][:]
            self.actions_history[0][:] = action_vector[:] 
        return self.actions_history

    def calculate_position_box(self, actions, xmin=0, xmax=224, ymin=0, ymax=224):
        """
            Prends l'ensemble des actions depuis le début et en génére les coordonnées finales de la boite englobante.
            Entrée :
                - Ensemble des actions sélectionnées depuis le début.
            Sortie :
                - Coordonnées finales de la boite englobante.
        """
        # Calcul des alpha_h et alpha_w mentionnées dans le papier
        alpha_h = self.alpha * (  ymax - ymin )
        alpha_w = self.alpha * (  xmax - xmin )
        real_x_min, real_x_max, real_y_min, real_y_max = 0, 224, 0, 224

        # Boucle sur l'ensemble des actions
        for r in actions:
            if r == 1: # Right
                real_x_min += alpha_w
                real_x_max += alpha_w
            if r == 2: # Left
                real_x_min -= alpha_w
                real_x_max -= alpha_w
            if r == 3: # Up 
                real_y_min -= alpha_h
                real_y_max -= alpha_h
            if r == 4: # Down
                real_y_min += alpha_h
                real_y_max += alpha_h
            if r == 5: # Bigger
                real_y_min -= alpha_h
                real_y_max += alpha_h
                real_x_min -= alpha_w
                real_x_max += alpha_w
            if r == 6: # Smaller
                real_y_min += alpha_h
                real_y_max -= alpha_h
                real_x_min += alpha_w
                real_x_max -= alpha_w
            if r == 7: # Fatter
                real_y_min += alpha_h
                real_y_max -= alpha_h
            if r == 8: # Taller
                real_x_min += alpha_w
                real_x_max -= alpha_w
        real_x_min, real_x_max, real_y_min, real_y_max = self.rewrap(real_x_min), self.rewrap(real_x_max), self.rewrap(real_y_min), self.rewrap(real_y_max)
        if real_x_min >= real_x_max :
            x_temp = real_x_min
            real_x_min = real_x_max
            real_x_max =  x_temp

        if real_y_min >= real_y_max :
            y_temp = real_y_min
            real_y_min = real_y_max
            real_y_max =  y_temp
        
        return [real_x_min, real_x_max, real_y_min, real_y_max]

    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates ):
        """
            Récupére parmis les boites englobantes vérité terrain d'une image celle qui est la plus proche de notre état actuel.
            Entrée :
                - Boites englobantes des vérités terrain.
                - Coordonnées actuelles de la boite englobante.
            Sortie :
                - Vérité terrain la plus proche.
        """
        max_iou = False
        max_gt = []
        for gt in ground_truth_boxes:
            iou = self.intersection_over_union(actual_coordinates, gt)
            if max_iou == False or max_iou < iou:
                max_iou = iou
                max_gt = gt
        return max_gt

    def predict_image(self, image, gt_boxes, plot=False):
        """
            Prédit la boite englobante d'une image
            Entrée :
                - Image redimensionnée.
            Sortie :
                - Coordonnées boite englobante.
        """

        # Passage du Q-Network en mode évaluation
        self.policy_net.eval()
        xmin = 0
        xmax = 224
        ymin = 0
        ymax = 224

        done = False
        all_actions = []
        self.actions_history = torch.ones((9,9))
        state = self.compose_state(image)
        original_image = image.clone()
        new_image = image

        steps = 0
        
        # Tant que le trigger n'est pas déclenché ou qu'on a pas atteint les 40 steps
        while not done:
            
            steps += 1
            action = self.select_action_model(state)
            all_actions.append(action)
            if action == 0:
                next_state = None
                new_equivalent_coord = self.calculate_position_box(all_actions)
                done = True
            else:
                # Mise à jour de l'historique
                self.actions_history = self.update_history(action)
                new_equivalent_coord = self.calculate_position_box(all_actions)            
                
                # Récupération du contenu de la boite englobante
                new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break            
                
                # Composition : état + historique des 9 dernières actions
                next_state = self.compose_state(new_image)
            
            if steps == 40:
                done = True
            
            # Déplacement au nouvel état
            state = next_state
            image = new_image
        
            if plot:
                show_new_bdbox(original_image, new_equivalent_coord, color='b', count=steps)

            for gt_box in gt_boxes:
                iou = self.intersection_over_union(gt_box, new_equivalent_coord)
                if iou>=0.5:
                    done = True
            
        

        # Génération d'un GIF représentant l'évolution de la prédiction
        if plot:
            #images = []
            tested = 0
            while os.path.isfile('media/movie_'+str(tested)+'.gif'):
                tested += 1
            # filepaths
            fp_out = "media/movie_"+str(tested)+".gif"
            images = []
            for count in range(1, steps+1):
                images.append(imageio.imread(str(count)+".png"))
            
            imageio.mimsave(fp_out, images)
            
            for count in range(1, steps):
                os.remove(str(count)+".png")
        
        return new_equivalent_coord, all_actions


    
    def evaluate(self, dataset):
        """
            Evaluation des performances du model sur un jeu de données.
            Entrée :
                - Jeu de données de test.
            Sortie :
                - Statistiques d'AP et RECALL.

        """
        ground_truth_boxes = []
        predicted_boxes = []
        print("Predicting boxes...")
        for key, value in dataset.items():
            image, gt_boxes = extract(key, dataset)
            bbox, action_list= self.predict_image(image)
            ground_truth_boxes.append(gt_boxes)
            predicted_boxes.append(bbox)
        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)
        print("Final result : \n"+str(stats))
        return stats

    def evaluate_iou(self, dataset):

        ground_truth_boxes = []
        predicted_boxes = []
        iou_scores = []
        action_list_all = []
        pos=0
        print("Predicting boxes...")
        for key, value in dataset.items():
            image, gt_boxes = extract(key, dataset)
            bbox, action_list = self.predict_image(image, gt_boxes)
            ground_truth_boxes.append(gt_boxes)
            predicted_boxes.append(bbox)

            for gt_box in gt_boxes:
                    
                    iou = self.intersection_over_union(gt_box, bbox)
                    print(f"Computed IOU: {iou}")
                    if iou<0:
                        iou = 0
                        #print(f"gt_box: {gt_box}, bbox: {bbox}")
                        #img_np = np.array(image)
                        #img_np = np.transpose(img_np, (1, 2, 0))
                        #plt.imshow(img_np)
                        #plt.show()
                    if iou >=0.5:
                        pos+=1
                    iou_scores.append(iou)
            print(action_list)
            print(len(action_list))
            action_list_all.append(len(action_list)+1)
                    
        print(f"pos%:{pos/len(iou_scores)}")
        average_iou = sum(iou_scores) / len(iou_scores)
        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)
        print("Final result : \n"+str(stats))
        #print(f"{len(iou_scores)}")
        print(f"Average IOU: {average_iou}")

        print(action_list_all)
        plt.hist(action_list_all)
        plt.show()
        
        #小於10的action數
        filtered_action_list_all = [action for action in action_list_all if action <= 10]
        print(filtered_action_list_all)
        plt.hist(filtered_action_list_all)
        plt.show()
        return stats

    def train(self, train_loader):
        """
            Fonction d'entraînement du modèle.
            Entrée :
                - Jeu de données d'entraînement.
        """
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0

        for i_episode in range(self.num_episodes):
            print("Episode "+str(i_episode))
            for key, value in  train_loader.items():
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                all_actions = []
        
                # Initialize the environment and state
                self.actions_history = torch.ones((9,9))
                state = self.compose_state(image)
                original_coordinates = [xmin, xmax, ymin, ymax]
                new_image = image

                done = False
                t = 0
                actual_equivalent_coord = original_coordinates
                new_equivalent_coord = original_coordinates
                while not done:
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth)
                    all_actions.append(action)
                    if action == 0:
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_trigger_reward(new_equivalent_coord,  closest_gt)
                        done = True

                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        

                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Move to the next state
                    state = next_state
                    image = new_image
                    # Perform one step of the optimization (on the target network)
                    self.optimize_model()
                    
            
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:
                self.EPS -= 0.18
            aa = 1
            self.save_network(i_episode,aa)

            print('Complete')

    def train_validate(self, train_loader, valid_loader):
        """
            Entraînement du modèle et à chaque épisode test de l'efficacité sur le jeu de test et sauvegarde des résultats dans un fichier de logs.
        """
        op = open("logs_over_epochs", "w")
        op.write("NU = "+str(self.nu))
        op.write("ALPHA = "+str(self.alpha))
        op.write("THRESHOLD = "+str(self.threshold))
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0
        for i_episode in range(self.num_episodes):  
            print("Episode "+str(i_episode))
            for key, value in  train_loader.items():
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                all_actions = []
        
                # Initialize the environment and state
                self.actions_history = torch.ones((9,9))
                state = self.compose_state(image)
                original_coordinates = [xmin, xmax, ymin, ymax]
                new_image = image
                done = False
                t = 0
                actual_equivalent_coord = original_coordinates
                new_equivalent_coord = original_coordinates
                while not done:
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth)
                    all_actions.append(action)
                    if action == 0:
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_trigger_reward(new_equivalent_coord,  closest_gt)
                        done = True

                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        
                        if False:
                            show_new_bdbox(original_image, ground_truth, color='r')
                            show_new_bdbox(original_image, new_equivalent_coord, color='b')
                            

                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Vers le nouvel état
                    state = next_state
                    image = new_image
                    # Optimisation
                    self.optimize_model()
                    
            stats = self.evaluate(valid_loader)
            op.write("\n")
            op.write("Episode "+str(i_episode))
            op.write(str(stats))
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:
                self.EPS -= 0.18
            aa = 2
            self.save_network(i_episode, aa)
            
            print('Complete')