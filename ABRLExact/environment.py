import random

import math
import numpy as np
from gymnasium import spaces



class DeepSea:
    def __init__(self, depth, starting_state=(0,0), goal_state=(-1,-1), penalty=None):
        self.env_name = 'GridWorld'
        self.n_cell = (depth, depth)
        self.starting_state = starting_state
        if goal_state == (-1, -1):
            goal_state = (self.n_cell[0] - 1, self.n_cell[1] - 1 )
        self.goal_state = goal_state
        self.treasure = 1 
        
        self.observation_space = spaces.Discrete(self.n_cell[0] * self.n_cell[1]) #total number of states
        self.action_space = spaces.Discrete(2)
        self.action_space_goal = spaces.Discrete(1)
        self.penalty = penalty
    
        self.trans_info, self.reward_info = self.initialise()
        self.state = self.starting_state
        self.done = False

        self.index = np.arange(self.observation_space.n*self.action_space.n).reshape((self.n_cell[0],self.n_cell[1],self.action_space.n))

    def initialise(self):
        #reward, first row is useless (TODO), first two dim - next_state, final dim - action that leads to the next state
        if self.penalty is None:
            reward_ =  np.round(np.array([-1, 1]) / ( 100 * self.n_cell[1] ), 4) #map action to reward for non goal states
        else:
            reward_ = np.array([-self.penalty, self.penalty])
        reward_mat = np.zeros(shape=(self.n_cell[0], self.n_cell[1], self.action_space.n)) #depth x depth x action_space
        reward_mat[:, :, 0] = reward_[0] #action 0 yields negative rewards
        reward_mat[:, :, 1] = reward_[1] #action 1 yields positive rewards
        reward_mat[-1, :, :] = 0 #final row yields zero rewards
        reward_mat[-2, -2, 0] = self.treasure #final box on diagonal gives final reward of 1
        
        #transition, final row is useless (TODO)
        gridworld = np.arange(self.observation_space.n).reshape((self.n_cell[0], self.n_cell[1])) #state numbers

        trans_mat = np.zeros((self.n_cell[0], self.n_cell[1], self.action_space.n, 2), dtype='int32') #real transition function - depth x depth x action_space x 2(left,right)
        for state_idx in gridworld.flat:
            row, col = np.argwhere(gridworld == state_idx)[0]
            for action, d in zip(range(self.action_space.n), [(1, 1), (1, -1)]):
                next_row = max(0, min(row + d[0], self.n_cell[0] - 1)) #always lead to next row
                next_col = max(0, min(col + d[1], self.n_cell[1] - 1)) #action 0 moves right, action 1 moves left
                s_prime = [next_row, next_col]
                trans_mat[row, col, action] = s_prime
                trans_mat[self.goal_state + (action, )] = self.goal_state #any goal action leads to goal
        return trans_mat, reward_mat

    def reset(self, state=None):
        if state is None:
            self.state = self.starting_state
        else:
            self.state = state

        self.done = False
        return self.state
    
    def reward(self, state, action, next_state):  # pylint: disable=unused-argument
        return self.reward_info[state + (action, )]
    
    def transition(self, state, action):
        return tuple(self.trans_info[state + (action, )])
    
    def transition_reward(self, state, action):
        next_state = tuple([int(i) for i in self.transition(state=state, action=action)])
        return self.reward(state=state, action=action, next_state=next_state), next_state

    def is_done(self, next_state):
        return next_state[0] == (self.n_cell[0]-1)
    
    def step(self, action, is_simulation=False, simulation_state=None):

        if is_simulation:
            assert simulation_state is not None
            state = simulation_state
        else:
            state = self.state

        assert self.done is False

        is_done = False
        reward, next_state = self.transition_reward(state=state, action=action)
        if self.is_done(next_state):
            is_done = True

        if not is_simulation:
            self.state = next_state
            self.done = is_done

        return next_state, reward, is_done
    
    def get_state_action_index(self, state, action):
        return self.index[state[0],state[1],action]
    
    def get_possible_actions(self, state, gym_space=True):
        if state[0] != (self.n_cell[0]-1):
            if gym_space is True:
                return self.action_space
            else:
                return list(range(self.action_space.n))
        else:
            if gym_space is True:
                return self.action_space_goal
            else:
                return list(range(self.action_space_goal.n))

    def get_all_states(self):
        states = [(i,j)  for i in range(self.n_cell[0]) for j in range(i+1)]
        return states


    def get_all_terminal_states(self):
        return [state for state in self.get_all_states() if state[0] == (self.n_cell[0]-1)]

    def nu(self, state, action):
        pos = (state[0])*(state[0]+1)+(state[1])*2+action + 1
        return pos
        
    def nu_inverse(self, pos):
        pos -= 1
        action = pos % 2
        ari_pos = (pos - action) // 2
    
        state0 = math.ceil((-1 + math.sqrt(1 + 8 * (ari_pos+1))) / 2) - 1
        prev_total = (state0) * (state0 + 1) // 2
        state1 = ari_pos + 1 - prev_total - 1
    
        return (state0, state1), action


def tabular_belief_update(suff_stat, state_o, action, reward, next_state_o):
    '''belief update'''
    suff_stat[state_o + (action,)][-1] = reward.item()    #last dim for reward
    suff_stat[state_o + (action,)][:-1] = next_state_o    #all but last dim for next state
    return suff_stat
    
class DeepSea_tabular_bamdp(DeepSea):
    def __init__(self, depth, starting_state=(0,0), goal_state=(-1,-1), reward_prior_var=1): #state of the form (state, suff_stat)
        super().__init__(depth=depth, starting_state=starting_state, goal_state=goal_state)
        self.reward_prior_var = reward_prior_var

    def initialise(self):
        return None, None

    def transition_reward(self, state, action):
        state_o, suff_stat = state

        reward = self.reward(state=state, action=action)
        next_state_o = self.transition_state_o(state=state, action=action)
        next_suff_stat = tabular_belief_update(suff_stat=suff_stat.copy(), state_o=state_o, action=action, reward=reward, next_state_o=next_state_o)

        next_state = (next_state_o, next_suff_stat)
        return reward, next_state

    def reward(self, state, action, next_state=None): #this has a slightly different structure as deepsea.Deepsea, but it's just a shift
        state_o, suff_stat = state
        if np.isnan(suff_stat[state_o + (action,)+ (0,)]):
            reward = self.sample_reward_prior(state_o=state_o, action=action)
        else:
            reward = suff_stat[state_o + (action,)][-1]
        return reward

    def transition_state_o(self, state, action):
        state_o, suff_stat = state
        if np.isnan(suff_stat[state_o + (action,)+ (0,)]):
            next_state_o = self.sample_transition_prior(state_o=state_o, action=action)
        else:
            next_state_o = tuple([int(i) for i in suff_stat[state_o + (action,)][:-1]])
        return next_state_o

    def is_done(self, next_state):
        return next_state[0][0] == (self.n_cell[0]-1)
    
    def get_possible_actions(self, state, gym_space=True):
        return super().get_possible_actions(state=state[0], gym_space=gym_space)

    def sample_transition_prior(self, state_o, action): # pylint: disable=unused-argument
        next_state_o_row = min(state_o[0]+1, self.n_cell[0]-1)
        next_state_o_col = max(0, min(state_o[1] + random.choice([-1,1]), self.n_cell[0]-1))
        return (next_state_o_row, next_state_o_col)
    
    def sample_reward_prior(self, state_o, action): # pylint: disable=unused-argument
        return np.random.randn(1)[0] * np.sqrt(self.reward_prior_var)

    def generate_key(self, state, action=None, state_only=False):
        state_o, suff_stat = state
        nan_mask = np.isnan(suff_stat[..., 0])
        suff_stat_binary = tuple(~nan_mask.flatten())
        if state_only:
            key = (tuple(state_o), suff_stat_binary)
        else:
            key = (tuple(state_o), suff_stat_binary, action)
        return key