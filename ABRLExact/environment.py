import random

import math
import numpy as np
from gymnasium import spaces

import torch



class DeepSea:
    def __init__(self, depth, starting_state=(0,0), goal_state=(-1,-1), deterministic_transition=True, randomised_actions=False, 
                 randomised_action_seed=None, penalty=None, sto_trans_prob=None):
        self.env_name = 'Deepsea'
        self.depth = depth
        self.starting_state = starting_state
        if goal_state == (-1, -1):
            goal_state = (depth - 1, depth - 1 )
        self.goal_state = goal_state
        self.deterministic_transition = deterministic_transition
        self.randomised_actions = randomised_actions
        self.penalty = penalty
        self.sto_trans_prob = sto_trans_prob

        if sto_trans_prob is None and deterministic_transition is False:
            self.sto_trans_prob = 1 / depth
        
        self.treasure = 1.
        if randomised_actions:
            np.random.seed(randomised_action_seed)
        
        self.observation_space = spaces.Discrete(depth * depth) #total number of states
        self.action_space = spaces.Discrete(2)
        self.action_space_goal = spaces.Discrete(1)

        self.trans_info, self.reward_info, self.action_map = self.initialise()
        self.index = np.arange(self.observation_space.n*self.action_space.n).reshape((depth, depth, self.action_space.n))
        self.state = self.starting_state
        self.done = False


    def initialise(self):

        if self.randomised_actions: # 1 means right=0; 0 means right=1
            action_mat = np.random.binomial(1, 0.5, size=(self.depth, self.depth))
        else:
            action_mat = np.ones([self.depth, self.depth], dtype=np.int32)

        if self.penalty is None:
            reward_ =  np.round(np.array([-1, 1]) / ( 100 * self.depth ), 4) #map action to reward for non goal states
        else:
            reward_ = np.array([-self.penalty, self.penalty])
        reward_mat = np.zeros(shape=(self.depth, self.depth, self.action_space.n)) #depth x depth x action_space

        trans_mat = np.zeros((self.depth, self.depth, self.action_space.n, 2), dtype='int32')
        for row in range(self.depth):
            for col in range(row+1):
                next_row = max(0, min(row + 1, self.depth))
                left_action = action_mat[row, col]
                right_action = 1 - left_action
                
                next_col_right = max(0, min(col + 1, self.depth))
                next_col_left = max(0, min(col - 1, self.depth))

                trans_mat[row, col, left_action] = [next_row, next_col_left]
                trans_mat[row, col, right_action] = [next_row, next_col_right]

                reward_mat[row, col, left_action] = reward_[1] #action right yields negative rewards
                reward_mat[row, col, right_action] = reward_[0] #action left yields negative rewards

        reward_mat[row, col, 1 - action_mat[-1, -1]] += self.treasure
                
        return trans_mat, reward_mat, action_mat

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
        if self.deterministic_transition or state[0] == self.depth - 1:
            return self.trans_info[state + (action, )]
        else:
            if random.random() < self.sto_trans_prob:
                return self.trans_info[state + (1 - action, )]
            else:
                return self.trans_info[state + (action, )]

    def transition_distribution(self, state, action):
        if self.deterministic_transition or state[0] == self.depth - 1:
            return {tuple(self.trans_info[state + (action, )].tolist()): 1., tuple(self.trans_info[state + (1 - action, )].tolist()): 0.}
        else:
            return {tuple(self.trans_info[state + (action, )].tolist()): 1 - self.sto_trans_prob, tuple(self.trans_info[state + (1 - action, )].tolist()): self.sto_trans_prob}
    
    def transition_reward(self, state, action):
        next_state = tuple([int(i) for i in self.transition(state=state, action=action)])
        return self.reward(state=state, action=action, next_state=next_state), next_state

    def is_done(self, next_state):
        return next_state[0] == self.depth
    
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
        if state[0] != self.depth:
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
        states = [(i,j)  for i in range(self.depth+1) for j in range(i+1)]
        return states

    def get_all_terminal_states(self):
        return [state for state in self.get_all_states() if state[0] == self.depth]

    def nu_vectorised(self, state, action):

        state = np.asarray(state)
        action = np.asarray(action)

        is_state_1d = (state.ndim == 1)

        if is_state_1d:
            state = state[np.newaxis, :]
            action = np.atleast_1d(action)

        pos = state[:,0] * (state[:,0] + 1) + state[:,1] * 2 + action + 1

        if is_state_1d:
            return pos.item()
        return pos
        
    def nu(self, state, action):
        return (state[0])*(state[0]+1)+(state[1])*2+action + 1
        
    def nu_inverse_vectorised(self, pos):
        pos = np.asarray(pos, dtype=np.int64)
        is_pos_0d = (pos.ndim == 0)

        if is_pos_0d:
            pos = np.atleast_1d(pos)
            
        (state0, state1), action = self.nu_inverse(pos)

        state = np.stack((state0, state1), axis=1)

        if is_pos_0d:
            return state[0], action.item()
        return state, action

    def nu_inverse(self, pos):
        pos -= 1
        action = pos % 2
        ari_pos = (pos - action) // 2
    
        state0 = math.ceil((-1 + math.sqrt(1 + 8 * (ari_pos+1))) / 2) - 1
        prev_total = (state0) * (state0 + 1) // 2
        state1 = ari_pos + 1 - prev_total - 1
    
        return (state0, state1), action

    def nu_torch(self, state, action):
        state = torch.as_tensor(state)
        action = torch.as_tensor(action)

        pos = state[..., 0] * (state[..., 0] + 1) + state[..., 1] * 2 + action + 1
        return pos


    def nu_inverse_torch(self, pos):
        pos = torch.as_tensor(pos)

        pos = pos - 1
        action = pos % 2
        ari_pos = torch.div(pos - action, 2, rounding_mode='floor')

        state0 = torch.ceil((-1 + torch.sqrt((1 + 8 * (ari_pos + 1)).float())) / 2).long() - 1
        prev_total = state0 * (state0 + 1) // 2
        state1 = ari_pos + 1 - prev_total - 1

        state = torch.stack((state0, state1), dim=-1)
        return state, action


class DeepSeaPyramid(DeepSea):
    def __init__(self, depth, starting_state=(0,0), goal_state=(-1,-1), deterministic_transition=True, randomised_actions=False, 
                 randomised_action_seed=None, sto_trans_prob=None):
        super().__init__(depth=depth, starting_state=starting_state, goal_state=goal_state, deterministic_transition=deterministic_transition, 
                         randomised_actions=randomised_actions, randomised_action_seed=randomised_action_seed, sto_trans_prob=sto_trans_prob)
        
    def initialise(self):

        reward_mat = np.zeros(shape=(self.depth, self.depth, self.action_space.n)) #depth x depth x action_space

        if self.randomised_actions:
            action_mat = np.random.binomial(1, 0.5, size=(self.depth, self.depth))
        else:
            action_mat = np.ones([self.depth, self.depth], dtype=np.int32)

        trans_mat = np.zeros((self.depth, self.depth, self.action_space.n, 2), dtype='int32')
        for row in range(self.depth):
            for col in range(row+1):
                next_row = max(0, min(row + 1, self.depth))
                left_action = action_mat[row, col]
                right_action = 1 - left_action
                
                next_col_right = max(0, min(col + 1, self.depth))
                next_col_left = max(0, min(col, self.depth))

                trans_mat[row, col, left_action] = [next_row, next_col_left]
                trans_mat[row, col, right_action] = [next_row, next_col_right]

            q, r = divmod(row, 2)
            if r == 0:
                reward_mat[row, q, :] = 1 / self.depth
            else:
                reward_mat[row, q, 1 - action_mat[row, q]] = 1 / self.depth # right
                reward_mat[row, q+1, action_mat[row, q]] = 1 / self.depth # left
                
        return trans_mat, reward_mat, action_mat


class DeepSeaSwirl(DeepSea):
    def __init__(self, depth, starting_state=(0,0), goal_state=(-1,-1), deterministic_transition=True, randomised_actions=False, 
                 randomised_action_seed=None, penalty=None, sto_trans_prob=None):
        super().__init__(depth=depth, starting_state=starting_state, goal_state=goal_state, deterministic_transition=deterministic_transition, 
                         randomised_actions=randomised_actions, randomised_action_seed=randomised_action_seed, penalty=penalty, sto_trans_prob=sto_trans_prob)
        
    def initialise(self):

        if self.penalty is None:
            reward_ =  np.round(np.array([-1, 1]) / ( 100 * self.depth ), 4) #map action to reward for non goal states
        else:
            reward_ = np.array([-self.penalty, self.penalty])

        reward_mat = np.zeros(shape=(self.depth, self.depth, self.action_space.n)) #depth x depth x action_space

        if self.randomised_actions:
            action_mat = np.random.binomial(1, 0.5, size=(self.depth, self.depth))
        else:
            action_mat = np.ones([self.depth, self.depth], dtype=np.int32)

        trans_mat = np.zeros((self.depth, self.depth, self.action_space.n, 2), dtype='int32')
        for row in range(self.depth):
            for col in range(row+1):
                next_row = max(0, min(row + 1, self.depth))
                left_action = action_mat[row, col]
                right_action = 1 - left_action
                
                next_col_right = max(0, min(col + 1, self.depth))
                next_col_left = max(0, min(col, self.depth))

                trans_mat[row, col, left_action] = [next_row, next_col_left]
                trans_mat[row, col, right_action] = [next_row, next_col_right]

                reward_mat[row, col, left_action] = reward_[1] #action right yields negative rewards
                reward_mat[row, col, right_action] = reward_[0] #action left yields negative rewards

                # swirl
                if row % 3 == 1:
                    if col % 3 == 0:
                        trans_mat[row, col, right_action] = [row, col + 1]
                        reward_mat[row, col, right_action] = reward_[0] * 3
                    elif col % 3 == 1:
                        trans_mat[row, col, left_action] = [row, col - 1]
                        reward_mat[row, col, left_action] = reward_[0] * 3
                elif row % 3 == 2:
                    if col % 3 == 0:
                        trans_mat[row, col, right_action] = [row, col + 1]
                        reward_mat[row, col, right_action] = reward_[0] * 3
                    elif col % 3 == 2:
                        trans_mat[row, col, left_action] = [row, col - 1]
                        reward_mat[row, col, left_action] = reward_[0] * 3

        reward_mat[row, col, 1 - action_mat[-1, -1]] += self.treasure

                
        return trans_mat, reward_mat, action_mat


class fourDMDP:
    def __init__(self):
        self.starting_state = 1
        self.state = self.starting_state
        self.initialise()
        self.reward_dict, self.transition_dict = self.initialise()
        self.done = False

    def initialise(self):
        reward_dict = dict()
        transition_dict = dict()

        reward_dict[(1,1)] = -2
        reward_dict[(1,2)] = -4
        reward_dict[(2,3)] = -1
        reward_dict[(2,4)] = -2
        reward_dict[(3,5)] = -1

        transition_dict[(1,1)] = 2
        transition_dict[(1,2)] = 3
        transition_dict[(2,3)] = 3
        transition_dict[(2,4)] = 4
        transition_dict[(3,5)] = 4

        return reward_dict, transition_dict

    def reset(self):
        self.state = self.starting_state
        self.done = False
        return self.state

    def is_done(self, next_state):
        return next_state == 4
    
    def step(self, action, is_simulation=False, simulation_state=None):

        if is_simulation:
            assert simulation_state is not None
            state = simulation_state
        else:
            state = self.state

        assert self.done is False

        done = False
        reward, next_state = self.transition_reward(state=state, action=action)
        if self.is_done(next_state):
            done = True

        if not is_simulation:
            self.state = next_state
            self.done = done

        return next_state, reward, done

    def transition_reward(self, state, action):
        reward = self.reward_dict[(state,action)]
        next_state = self.transition_dict[(state,action)]
        return reward, next_state

    def transition_distribution(self, state, action):
        return {self.transition_dict[(state,action)]: 1.}

    def get_all_states(self):
        return [1,2,3,4]

    def get_possible_actions(self, state, gym_space=False):
        if state == 1:
            if gym_space is True:
                return spaces.Discrete(2)
            else:
                return [1,2]
        if state == 2:
            if gym_space is True:
                return spaces.Discrete(2)
            else:
                return [3,4]
        if state == 3:
            if gym_space is True:
                return spaces.Discrete(1)
            else:
                return [5]
        if state == 4:
            if gym_space is True:
                return spaces.Discrete(1)
            else:
                return [6]

    def get_all_terminal_states(self):
        return [4]

    def nu(self, state, action): # pylint: disable=unused-argument
        return action

    def nu_vectorised(self, state, action):
        return self.nu(state=state, action=action)

    def nu_inverse(self, pos): 
        pos = np.asarray(pos)
        is_scalar = (pos.ndim == 0)

        conds = [np.isin(pos, [1,2]), np.isin(pos, [3,4]), pos == 5]

        choices = [1, 2, 3]

        state = np.select(conds, choices)
        action = pos

        if is_scalar:
            return state.item(), action.item()
        return state, action