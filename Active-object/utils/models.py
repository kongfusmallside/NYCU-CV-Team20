import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights

"""
    Modèle d'extraction de caractéristiques, ici en faisant appel à VGG-16.
"""
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__() # recopier toute la partie convolutionnelle
        vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg16.eval() # to not do dropout
        self.features = list(vgg16.children())[0] 
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])
    def forward(self, x):
        x = self.features(x)
        return x
    
"""
    Architecture du Q-Network comme décrite dans l'article.
"""
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear( in_features= 81 + 25088, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=9)
        )
    def forward(self, x):
        return self.classifier(x)
    

class DuelingDQN(nn.Module):
    def __init__(self, input_dim=81+25088, outputs=9, hidden_dim1=1024, hidden_dim2=512):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim1),
            nn.Dropout(0.2)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim2),
            nn.Linear(hidden_dim2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim2),
            nn.Linear(hidden_dim2, outputs)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
