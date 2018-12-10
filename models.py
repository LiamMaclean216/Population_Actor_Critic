import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

class Creature(nn.Module):
    def __init__(self,input_size,output_size):
        super(Creature, self).__init__()
        self.input_size =input_size
        self.output_size = output_size
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, output_size)
    
    def forward(self, x):
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=1, padding=0),  
            #nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
            
        self.layer2 = nn.Sequential(    
            nn.Conv1d(8, 16, 5, stride=1, padding=0),  
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        
        self.layer3 = nn.Sequential(    
            nn.Conv1d(16, 32, 5, stride=1, padding=0),  
            #nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        self.layer4 = nn.Sequential(    
            nn.Conv1d(32, 16, 5, stride=1, padding=0),  
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        
        self.layer5 = nn.Linear(5856, 256)
        self.layer6 = nn.Linear(256, 16)
        self.layer7 = nn.Sequential(
           # nn.Dropout(0.2),
            nn.Linear(16, 2))
        
        
    def forward(self, out):
        out = out.unsqueeze(1)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),out.size(1)*out.size(2))
        
        out = self.layer5(out)
        out_latent = self.layer6(out)
        out = self.layer7(out_latent)
        
        #confidence = nn.Sigmoid()(nn.Softplus()(out[...,1]).unsqueeze(-1))
        confidence = nn.Sigmoid()(out[...,1]).unsqueeze(-1) * 4
        prediction = nn.Tanh()(out[...,0]).unsqueeze(-1) 
        
        #confidence = nn.Sigmoid()(out[...,1]).unsqueeze(-1) * 5
        #prediction = nn.Tanh()(out[...,0]).unsqueeze(-1) * 5
        
        out = torch.cat([prediction,confidence],-1)
        
        return out,out_latent
    
class Generator(nn.Module):
    def __init__(self,creature_size,device,population_size):
        super(Generator, self).__init__()
        self.device = device
        self.population_size = population_size
        self.creature_size = creature_size
        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(386, 256, 4, stride=1, padding=0),  
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),)
            #nn.MaxPool1d(2, stride=1))
            
        self.layer2 = nn.Sequential(    
            nn.ConvTranspose1d(256, 128, 4, stride=1, padding=0),  
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),)
            #nn.MaxPool1d(2, stride=1))
        
        self.hidden_size = 300
        self.n_layers = 1
        self.gru = nn.GRU(896, self.hidden_size, self.n_layers, bidirectional=True)
        self.hidden = None
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=1, padding=0),  
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
            
        self.layer4 = nn.Sequential(    
            nn.Conv1d(8, 16, 5, stride=1, padding=0),  
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        
        self.layer5 = nn.Linear(9440, creature_size)
        
    def forward(self,x):
        
        out = self.layer1(x.unsqueeze(-1))
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(self.population_size,out.size(0)//self.population_size,out.size(1)*out.size(2))
        out, self.hidden = self.gru(out,self.hidden)
        out = out.view(out.size(0)*out.size(1),1,out.size(2))
        #print(out.shape)
        out = self.layer3(out)
        out = self.layer4(out)
        
        #print(out.shape)
        out = out.view(out.size(0),out.size(1)*out.size(2))
        
        out = self.layer5(out)
                
        
        #out = torch.tanh(out) * lr
        
        return out,out
