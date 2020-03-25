# -*- coding: utf-8 -*-

import numpy as np
from visualization import ImageResult, show_video
import time
import gym
from gym import spaces


class Environment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    shape = 7
    vision = 2
    hunters = 0
    preys = 0
    agents = []
    actions = []
    step_nb = 0
    
    action_to_delta = {
    0:np.array([1,0]),#droite
    1:np.array([0,-1]),#bas
    2:np.array([-1,0]),#gauche
    3:np.array([0,1]),#haut
    4:np.array([0,0])
    }
    
    def __init__(self,shape=shape,vision=vision, nb_hunters = 4,positions = None, actions = list(range(5))):
        super(Environment,self).__init__()
        self.shape = shape
        self.nb_hunters = nb_hunters
        self.actions = actions
        self.step_nb = 0
        self.init_positions = positions
        self.vision = vision
        self.observation_space = spaces.Box(low = np.array([[0,0],[0,0],[0,0],[0,0],[0,0]]),high = np.array([[shape-1,shape-1],[shape-1,shape-1],[shape-1,shape-1],[shape-1,shape-1],[shape-1,shape-1]]),dtype=np.int16)
        self.action_space = spaces.Discrete(625)
        if positions == None:
            #hunters are put at the four corners of the environment and the prey at the center
            positions = [np.array([0,0]),np.array([0,self.shape-1]),np.array([self.shape-1,0]),np.array([self.shape-1, self.shape-1]),np.array([self.shape//2, self.shape//2])]
        
        self.hunters = []
        for i in range(nb_hunters):
            self.hunters.append(Agent(1,positions[i]))
        self.prey = Agent(0, positions[self.nb_hunters])
    
    def move_prey(self, p_still=0.5):
        u = np.random.rand()
        if u<p_still:
            return 4 #the prey remains still
        else:
            possible_actions = self.select_possible_actions(self.prey.position)
            return np.random.choice(possible_actions[1:]) #so that we don't consider standing still
        

    def select_possible_actions(self, position): 
        p = []
        p.append(4)
        if position[0]>0 :
            p.append(2)
        if position[0] < self.shape -1 :
            p.append(0)
        if position[1] > 0 :
            p.append(1)
        if position[1] < self.shape -1:
            p.append(3)
        return p
            
    def step(self,actions):
        
        pos_prey= self.prey.position + self.action_to_delta[self.move_prey()]
        pos_hunters = []
        
        for i in range(self.nb_hunters):
            a = self.hunters[i]
            possible_actions = self.select_possible_actions(a.position)
            if actions[i] in possible_actions :
                pos_hunters.append(self.hunters[i].position + self.action_to_delta[actions[i]])
            else : 
                pos_hunters.append (np.array(self.hunters[i].position))
                
        
        
        #check for incompatible behaviour
        moving = [True, True , True, True, True]
        conflit = True
        s = 1
        while conflit and s<5:
            flag = False
            for i in range(self.nb_hunters):
                if pos_prey[0] == pos_hunters[i][0] and pos_prey[1] == pos_hunters[i][1]:
                    moving[0]=False
                    pos_prey = self.prey.position
                    moving[i+1] = False
                    pos_hunters[i] = self.hunters[i].position
                    flag = True
                for j in range(i+1,self.nb_hunters):
                    if pos_hunters[i][0] == pos_hunters[j][0] and pos_hunters[i][1] == pos_hunters[j][1] :
                        moving[i+1]=False
                        moving[j+1]=False
                        pos_hunters[i] = self.hunters[i].position
                        pos_hunters[j] = self.hunters[j].position
                        flag = True
            if flag == False :
                conflit = False
            s+=1
                
                        
                
                
        #update the agents positions if needed
        if moving[0] :
            self.prey.position = pos_prey
        for i in range(self.nb_hunters):
            if moving[i+1] :
                self.hunters[i].position = pos_hunters[i]
        
        self.step_nb += 1
        
        return self.get_all_positions(), self.reward(), self.done(), {}
        
    def get_all_positions(self) :
        positions = [self.prey.position]
        for i in range(self.nb_hunters):
            positions.append(self.hunters[i].position)
        return np.array(positions)
    
    def reset(self):
        if self.init_positions == None:
            positions = [np.array([0,0]),np.array([0,self.shape-1]),np.array([self.shape-1,0]),np.array([self.shape-1, self.shape-1]),np.array([self.shape//2, self.shape//2])]
            
        else:
            positions = self.init_positions
            
        for i in range(self.nb_hunters):
            self.hunters[i].position=positions[i]
            self.prey.position=positions[-1]
    
    def render(self, mode='human', close=False):
        print(self.reward)
    
    def reward(self):
        vision = self.vision
        rewards = [0,0,0,0]
        for i in range(self.nb_hunters):
            if np.abs(self.hunters[i].position[0]-self.prey.position[0])+np.abs(self.hunters[i].position[1]-self.prey.position[1]) ==1: #the hunter is next to the prey
                rewards[i]+=10
                
        if sum(rewards) == 40: #the 4 hunters have circled the prey
            return [100,100,100,100]
        
        nb_possible_actions_prey = len(self.select_possible_actions(self.prey.position)) #self.select_possible_actions indicates only takes the walls into account
        if sum(rewards) == 30 and nb_possible_actions_prey==4: #there are 3 hunters around the prey + 1 wall
            return [100*int(rewards[i]!=0) for i in range(4)]
        
        if sum(rewards) == 20 and nb_possible_actions_prey==3: #there are 2 huters around the prey + 2 walls
            return [100*int(rewards[i]!=0) for i in range(4)]
        
        if sum(rewards) >10 :
            for i in range(self.nb_hunters):
                rewards[i] = rewards[i]+int(rewards[i]!=0) * 5 * sum(rewards)//10 
        
        for i in range(self.nb_hunters):
            if np.abs(self.hunters[i].position[0]-self.prey.position[0])<= vision and np.abs(self.hunters[i].position[1]-self.prey.position[1]) <=vision: #the hunter sees the prey
                rewards[i]+=1
        
        return rewards
    
    def done(self):
        for moves in self.select_possible_actions(self.prey.position)[1:]:
            if not list(self.prey.position + self.action_to_delta[moves]) in [list(i.position) for i in self.hunters]:
                return False
        return True
    
    def show(self):
        picture = ImageResult(self.shape,30,self.vision)
        return picture.draw_obs(self.get_all_positions())
        #picture.show()

        

class Agent:
    
    role = None
    position = None
    decision_function = None
    
    def __init__(self,role,position,decision_function = lambda _,l : np.random.choice(l)):
        self.role = role
        self.position = position
        self.decision_function = decision_function
        
    def decision(self,voisions,possible_actions):
        return self.decision_function(voisions,possible_actions) # Par défaut c'est np.random.choice
    

def demo() :
    env = Environment()
    images =[env.show()]
    print(env.observation_space.low)
    for i in range(100):
        env.step(np.random.randint(5, size=4))
        images.append(env.show())
    show_video(images,0)
    return