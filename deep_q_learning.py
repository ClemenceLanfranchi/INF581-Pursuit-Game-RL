# -*- coding: utf-8 -*-
"""
Parts of the code were inspired from: https://keon.github.io/deep-q-learning/
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from env import Environment
from visualization import ImageResult, show_video
from tensorflow import keras
from collections import deque
from replay_buffer import ReplayBuffer
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

class DeepQ:
    
    def __init__(self, environnment):
        self.state_size = 16
        self.action_size = len(environnment.actions)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.tau = 1
        self.tau_inc = 0.01
        self.init_tau = 1
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def softmax(self,q):
        assert self.tau >= 0.0
        q_tilde = q - np.max(q)
        factors = np.exp(self.tau * q_tilde)
        return factors / np.sum(factors)
    

    def act(self, state): # Implementes softmax
        act_values = self.model.predict(state)
        prob_a = self.softmax(act_values[0])
    
        cumsum_a = np.cumsum(prob_a)
        return np.where(np.random.rand() < cumsum_a)[0][0]  # When this line is removed implementes epsilon greedy
        
        
        if np.random.rand() <= self.epsilon: 
            return np.random.randint(self.action_size)
        if state[0][0]==-100:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibach = random.sample(self.memory,batch_size)
        x = np.array([[]])
        y = np.array([[]])
        for state, action, reward, next_state, done in minibach:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            if np.size(x)==0:
                x = state
            else:
                x = np.concatenate((x,state))
            if np.size(y)==0:
                y = target_f
            else:
                y = np.concatenate((y,target_f))
        self.model.fit(x, y, epochs=20,batch_size = batch_size, verbose=0)
        self.tau = self.init_tau + i_episode * self.tau_inc

def surrounding_state(env, hunter):
    #positions of the hunters and prey and walls if they are visible
    #hunter : index of hunter in list of hunters
    vision = env.vision
    shape = env.shape
    state = []
    pos_hunter = env.hunters[hunter].position
    pos_prey = env.prey.position
    
    #position of the prey
    if np.abs(pos_prey[0]-pos_hunter[0])<=vision and np.abs(pos_prey[1]-pos_hunter[1])<=vision :
        state.append(pos_prey - pos_hunter)
        state.append([0])
    else : 
        state.append(np.array([0,0]))
        state.append([1])
        

    #position of the hunters
    nbh_vision = 0
    relative_positions=[]
    for i in range (env.nb_hunters):
        if i == hunter:
            continue
        pos_other_hunter = env.hunters[i].position
        if np.abs(pos_other_hunter[0]-pos_hunter[0])<=vision and np.abs(pos_other_hunter[1]-pos_hunter[1])<=vision  :
            relative_positions.append(pos_other_hunter - pos_hunter)
            nbh_vision +=1
            
    np.sort(relative_positions, axis=0) #so that the state is not dependent on the order of the hunters
    
    
    for i in range(nbh_vision):
        state.append(relative_positions[i])
        state.append([0])   # This state signifies the pray is visible for hunter i
    
    for i in range(env.nb_hunters-nbh_vision-1):
        state.append(np.array([0, 0]))  
        state.append([1])   # This state signifies the pray is not visible for hunter i

    #position of the walls       
    if pos_hunter[0]<vision :
        pos_wall_x = -1-pos_hunter[0]
    elif pos_hunter[0]>=shape - vision :
        pos_wall_x = shape-pos_hunter[0]
    else :
        pos_wall_x = 0
    if pos_hunter[1]<vision :
        pos_wall_y = -1-pos_hunter[1]
    elif pos_hunter[1]>=shape - vision :
        pos_wall_y = shape-pos_hunter[1]
    else :
        pos_wall_y = 0   
    
    state.append(np.array([pos_wall_x,pos_wall_y]))

    # These lines do the same work as whatis done previously to signify whether a wall is visible
    
    if pos_wall_x == 0:
        state.append([1])
    else:
        state.append([0])
    if pos_wall_y == 0:
        state.append([1])
    else:
        state.append([0])
    r = []
    for i in state:
        r += list(i)
    return r
    

def visions(env):
    #array of the surrounding states of all hunters
    visions = []
    for i in range(env.nb_hunters):
        visions.append(surrounding_state(env,i))
    return np.array(visions)
        
if __name__ == "__main__":

    # initialize gym environment and the agent
    env = Environment()
    agent = DeepQ(env)
    
    n_episode = 1000
    print("n_episode ", n_episode)
    max_horizon = 300
    
    rewards_list = []
    successes = []
    nb_steps = []

    # Iterate the game
    for i_episode in range(n_episode):

        # reset state in the beginning of each game
        env.reset()
        states = visions(env)
        
        images =[env.show()]
        rewards_episode = []

        for i_step in range(max_horizon):
            actions = []
            # Decide action
            for i in range(env.nb_hunters):
                state = np.reshape(states[i], [1, agent.state_size])
                actions.append(agent.act(state))

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            obs_prime, rewards, done, info = env.step(actions)
            images.append(env.show())
            rewards_episode.append(sum(rewards))
            states_prime = visions(env)
            
            # memorize the previous state, action, reward, and done
            for i in range(env.nb_hunters):
                state = np.reshape(states[i], [1,agent.state_size])
                next_state = np.reshape(states_prime[i], [1, agent.state_size])
                agent.memorize(state, actions[i], rewards[i], next_state, done)

            # make next_state the new current state for the next frame.
            
            states = states_prime.copy()

            # done becomes True when the game ends
            rewards_list.append(np.sum(rewards_episode))
            if done:
                # print the score and break out of the loop
                successes.append(1)
                nb_steps.append(i_step)
                break
            
            if i_step == max_horizon-1 :
                successes.append(0)
                nb_steps.append(i_step)
                
            
        
        if (i_episode+1)%100==0:
            show_video(images, i_episode)
            
        print("We are currently at episode ",i_episode," which took ",i_step," steps")

        # train the agent with the experience of the episode
        agent.replay(256)
    
    plt.figure(0)
    plt.plot([np.mean(rewards_list[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Policy with {0} and {1}".format("Deep_Q", "softmax"))
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average rewards")
    plt.savefig("rewards.png")
    plt.show()
    
    plt.figure(1)
    plt.plot([np.mean(successes[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Policy with {0} and {1}".format("Deep_Q", "softmax"))
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average success rate")
    plt.savefig("sucess_rate.png")
    plt.show()
    
    plt.figure(2)
    plt.plot([np.mean(nb_steps[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Policy with {0} and {1}".format("Deep_Q", "softmax"))
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average time steps")
    plt.savefig("time_step.png")
    plt.show()
