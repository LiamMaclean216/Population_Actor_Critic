import torch
import torch.nn as nn
import gym
from itertools import count
import numpy as np
import random
import torch.nn.functional as F
from models import Creature

#turn vector into model parameters
def set_params(model,data):
    idx = 0
    for p in model.parameters():
        view = data[idx:idx+p.numel()].view(p.shape)
        p.data = view
        idx+=p.numel()
    return model

#get model parameters as vector
def get_params(model):
    params = []
    for p in model.parameters():
        view = p.view(p.numel())
        params.append(view)
    params = torch.cat(params, dim=0)
    return params

#measure creature fitness
def measure_fitness(creature,env,device,discrete_actions,min_reward,render = False,max_steps = 1000):
    observation = env.reset()
    
    #creature fitness is cumulative reward in simulation
    total_reward = 0
    i = 0
    while True:
        if (i >= max_steps and max_steps > 0) or total_reward < min_reward:
            break
            
        if render:
            env.render()
            
        #convert observation into tensor
        obs = torch.from_numpy(observation).type('torch.FloatTensor').to(device)
       
        #get action
        if discrete_actions:
            action = creature(obs)
            sample = (obs,action)
            action = action.max(-1)[1].item()
        else:
            action = creature(obs)
            sample = (obs,action)
            action = action.detach().cpu().numpy()
            
        observation, reward, done, _ = env.step(action)
        
        total_reward += reward
        
        if done:
            break
        i+=1
    return total_reward

#measure fitness of entire population and return scores
def measure_population_fitness(population,env,device,discrete_actions,min_reward,max_steps = 1000):
    scores = []

    for p in population:
        fitness = measure_fitness(p,env,device,discrete_actions,min_reward,max_steps = max_steps)
        scores.append(fitness)
    return np.array(scores)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    