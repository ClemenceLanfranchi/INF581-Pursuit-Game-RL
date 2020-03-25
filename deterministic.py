# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
from env import Environment
from visualization import ImageResult, show_video
from collections import deque

 

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
    else : 
        state.append(np.array([-100,-100]))
        
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
    
    for i in range(env.nb_hunters-nbh_vision-1):
        state.append(np.array([-100, -100]))
            
    #position of the walls       
    if pos_hunter[0]<vision :
        pos_wall_x = -1-pos_hunter[0]
    elif pos_hunter[0]>=shape - vision :
        pos_wall_x = shape-pos_hunter[0]
    else :
        pos_wall_x = -100
    if pos_hunter[1]<vision :
        pos_wall_y = -1-pos_hunter[1]
    elif pos_hunter[1]>=shape - vision :
        pos_wall_y = shape-pos_hunter[1]
    else :
        pos_wall_y = -100    
    
    state.append(np.array([pos_wall_x,pos_wall_y]))
    r = []
    for i in state:
        r += list(i)
    
    return r

def visions(env):
    #array of the surrounding states of all hunters
    visions = []
    for i in range(env.nb_hunters):
        visions.append(surrounding_state(env,i))
    return visions


#Deterministic policy
def act_with_deterministic(s):
    if s[0] == -100:
        a = np.random.randint(5)
    else:
        a = np.random.choice(det[s[0]+2][s[1]+2])
    return a

#Deterministic policy table : actions to take depending on the input (position of the prey only)
det = [[[1,2],[2],[2],[2],[2,3]],[[1],[1,2],[2],[2,3],[3]],[[1],[1],[4],[3],[3]],[[1],[0,1],[0],[0,3],[3]],[[0,1],[0],[0],[0],[0,3]]]
#Caution : Works only for a visibility of radius 2

def main():      
    env = Environment()
    
    n_a = 5
    
    # Experimental setup
    n_episode = 1000
    print("n_episode ", n_episode)
    max_horizon = 300
    
    
    rewards_list = []
    successes = []
    nb_steps = []
    
    for i_episode in range(n_episode):
        
        env.reset()
        states = visions(env)
        actions = []
        
        # Select the first action in this episode
        for i in range(env.nb_hunters):
            actions.append(act_with_deterministic(states[i]))
        
        images =[env.show()]
        rewards_episode = []
        
        for i_step in range(max_horizon):

            # Act
            obs_prime, rewards, done, info = env.step(actions)
            images.append(env.show())
            rewards_episode.append(sum(rewards))
            states_prime = visions(env)         

            # Select an action
            actions_prime = []
            for i in range(env.nb_hunters):
                actions_prime.append(act_with_deterministic(states_prime[i]))

                
            # Transition to new state
            states = states_prime.copy()
            actions = actions_prime.copy()
            
            if done:
                successes.append(1)
                nb_steps.append(i_step)
                break
            
            if i_step == max_horizon-1 :
                successes.append(0)
                nb_steps.append(i_step)
        
        rewards_list.append(np.sum(rewards))
        
        if (i_episode+1)%100==0:
            show_video(images, i_episode)

    #Results plot
    plt.figure(0)
    plt.plot([np.mean(rewards_list[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Deterministic policy")
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average rewards")
    plt.show()
    
    plt.figure(1)
    plt.plot([np.mean(successes[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Deterministic policy")
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average success rate")
    plt.show()
    
    plt.figure(2)
    plt.plot([np.mean(nb_steps[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Deterministic policy")
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average time steps")
    plt.show()
        
if __name__ == "__main__":

    main()
        
        
