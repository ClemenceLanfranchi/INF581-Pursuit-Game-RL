# -*- coding: utf-8 -*-
"""
Some parts of the code were inspired from the Lab6
on Reinforcement Learning 2
"""

import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
from env import Environment
from visualization import ImageResult, show_video
from collections import deque
import copy
 
# Meta parameters for the RL agent
alpha = 0.05
tau = init_tau = 1
tau_inc = 0.01
gamma = 0.99
epsilon = 0.5
epsilon_decay = 0.999

SARSA = "SARSA"
Q_LEARNING = "Q_learning"
EPSILON_GREEDY = "epsilon-greedy"
SOFTMAX = "softmax"

# Choose methods for learning and exploration
rl_algorithm = Q_LEARNING
explore_method = EPSILON_GREEDY

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
    return copy.deepcopy(r)

# The four following functions are used to reduce 
# the state space by considering the symetries of the problem
    
def symmetry(s):
    first = -1
    wall = -1
    for i in range(0,len(s),2):
        if s[i] != -100 or s[i+1] != -100:
            first = i
            if s[i] == -100:
                wall = 0
            elif s[i+1] == -100:
                wall = 1
            break
    if first == -1 :
        return [s,0,0]
    else :
        if wall ==-1 :
            x,y =s[first],s[first+1]
            rot = 0
            sy = 0
            while x<0 or y<0 :
                s =rotate(s)
                rot+=1
                x,y =s[first],s[first+1]
            if x>y :
                s = sym(s)
                sy =1
        elif wall ==0:
            rot = 0
            sy=0
            if s[first+1]<0:
                s = rotate(rotate(s))
                rot =2   
        else:
            sy=0
            rot=0
            if s[first]<0:
                s = rotate(rotate(s))
                rot =2
    return [s, rot, sy]
    
    
def rotate(state) :
    new_state = copy.deepcopy(state)
    for i in range(0,len(state),2):  
        new_state[i],new_state[i+1] = state[i+1],-state[i]
        if new_state[i]==100:
            new_state[i]=-100
        if new_state[i+1]==100:
            new_state[i+1]=-100
    return new_state

def sym(state):
    new_state = copy.deepcopy(state)
    for i in range(0,len(state),2):  
        new_state[i],new_state[i+1] = state[i+1],state[i]
    return new_state

def new_action(action, rot, sym):
    
    if sym == 1:
        new = 3-action
    else :
        new = action
    new= (new - rot)%4
    return new


def visions(env):
    #array of the surrounding states of all hunters
    visions = []
    for i in range(env.nb_hunters):
        visions.append(symmetry(surrounding_state(env,i)))
    return visions

# Compute Q-Learning update
def q_learning_update(q,s,a,r,s_prime):
    a_max = np.argmax(q[s_prime])
    td = r + gamma * q[s_prime][a_max] - q[s][a]
    return q[s][a] + alpha * td

# Compute SARSA update
def sarsa_update(q,s,a,r,s_prime,a_prime):
    td = r + gamma * q[s_prime][a_prime] - q[s][a]
    return q[s][a] + alpha * td

def softmax(q):
    assert tau >= 0.0
    q_tilde = q - np.max(q)
    factors = np.exp(tau * q_tilde)
    return factors / np.sum(factors)

# Act with softmax
def act_with_softmax(s, q):
    prob_a = softmax(q[s])
    
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]

# Act with epsilon greedy
def act_with_epsilon_greedy(s, q):
    a = np.random.choice(np.argwhere(q[s]==np.max(q[s])).flatten())
    if np.random.rand() < epsilon:
        a = np.random.randint(5)
    return a


def main():
    global epsilon
    global tau
    
    env = Environment()
    
    n_a = 5
    
    # Init Q-table as a nested dictionary that maps state -> (action -> action-value). 
    q_table = defaultdict(lambda: np.zeros(n_a))
    
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
        if explore_method == EPSILON_GREEDY:
            for i in range(env.nb_hunters):
                actions.append(act_with_epsilon_greedy(tuple(states[i][0]), q_table))
        elif explore_method == SOFTMAX:
            for i in range(env.nb_hunters):
                actions.append(act_with_softmax(tuple(states[i][0]), q_table))
        else:
            raise ValueError("Wrong Explore Method:".format(explore_method))
        
        images =[env.show()]
        rewards_episode = []
        
        for i_step in range(max_horizon):

            # Act
            new_actions = [new_action(actions[i],states[i][1],states[i][2]) for i in range(len(actions))]
            obs_prime, rewards, done, info = env.step(new_actions)
            images.append(env.show())
            rewards_episode.append(sum(rewards))
            states_prime = visions(env)         
            
            # Select an action
            actions_prime = []
            if explore_method == SOFTMAX:
                for i in range(env.nb_hunters):
                    actions_prime.append(act_with_softmax(tuple(states_prime[i][0]), q_table))
            elif explore_method == EPSILON_GREEDY:
                for i in range(env.nb_hunters):
                    actions_prime.append(act_with_epsilon_greedy(tuple(states_prime[i][0]), q_table))
            else:
                raise ValueError("Wrong Explore Method:".format(explore_method))

            # Update a Q value table
            if rl_algorithm == SARSA:
                for i in range(env.nb_hunters):
                    q_table[tuple(states[i][0])][actions[i]] = sarsa_update(q_table,tuple(states[i][0]),actions[i],rewards[i],tuple(states_prime[i][0]),actions_prime[i])

            elif rl_algorithm == Q_LEARNING:
                for i in range(env.nb_hunters):
                    q_table[tuple(states[i][0])][actions[i]] = q_learning_update(q_table,tuple(states[i][0]),actions[i],rewards[i],tuple(states_prime[i][0]))
            else:
                raise ValueError("Wrong RL algorithm:".format(rl_algorithm))
                
            # Transition to new state
            states = copy.deepcopy(states_prime)
            actions =  copy.deepcopy(actions_prime)
            
            if done:
                successes.append(1)
                nb_steps.append(i_step)
                break
            
            if i_step == max_horizon-1 :
                successes.append(0)
                nb_steps.append(i_step)
        
        rewards_list.append(np.sum(rewards))
        
        if (i_episode)%100==0:
            show_video(images, i_episode)
            print(len(q_table)) #to get an idea of the number of observed states

        if (i_episode)%100==0:
            # Schedule for epsilon
            if epsilon > 0.01:
                epsilon = epsilon * epsilon_decay
            # Schedule for tau
            tau = init_tau + i_episode * tau_inc

    plt.figure(0)
    plt.plot([np.mean(rewards_list[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Policy with {0} and {1}".format(rl_algorithm, explore_method))
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average rewards")
    plt.show()
    
    plt.figure(1)
    plt.plot([np.mean(successes[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Policy with {0} and {1}".format(rl_algorithm, explore_method))
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average success rate")
    plt.show()
    
    plt.figure(2)
    plt.plot([np.mean(nb_steps[i*100:(i+1)*100]) for i in range(n_episode//100)])
    plt.title("Policy with {0} and {1}".format(rl_algorithm, explore_method))
    plt.xlabel("Number of episodes (x100)")
    plt.ylabel("Average time steps")
    plt.show()
        
if __name__ == "__main__":

    main()
        
        
