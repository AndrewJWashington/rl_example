# adapted from https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.optimizers import Adam

from collections import deque

class DQN:
    def __init__(self):
        self.state_shape = (4, 4, 1)
        self.num_actions = 4
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        #state_shape  = self.env.observation_space.shape 
        #model.add(Conv2D(filters=6, input_shape=self.state_shape,   
        #                 kernel_size=(2,2), strides=(1,1), padding="valid",
        #                 activation = "relu"))  # convolutional input
        #model.add(Flatten())
        model.add(Dense(24, input_dim=self.state_shape[0]*self.state_shape[1],
                        activation="relu"))  #dense input
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.num_actions))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.floor(np.random.rand() * self.num_actions)
        #return np.argmax(self.model.predict(state.reshape(4, 4, 1, 1))[0])
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        #print('got samples')
        for sample in samples:
            state, action, reward, new_state, done = sample
            #target = self.target_model.predict(state.reshape(4, 4, 1, 1))
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                #Q_future = max(self.target_model.predict(new_state.reshape(4, 4, 1, 1))[0])
                #print(target)
                #print(target[0])
                #print(action)
                #print(target[0][int(action)])

                # will need some work here to store and access the 4x1 output vector
                # possibly store as dictionary instead of deque
                target[0][int(action)] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def get_reward(cur_state, action):
    # tl means top left
    tl_mask = np.array([[1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
    tr_mask = np.array([[0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
    bl_mask = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 0, 0]])
    br_mask = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1]])
    
    if action == 0 and np.any(tl_mask * cur_state.reshape(4, 4)):
        return 0.8
    if action == 1 and np.any(tr_mask * cur_state.reshape(4, 4)):
        return 0.8
    if action == 2 and np.any(bl_mask * cur_state.reshape(4, 4)):
        return 0.8
    if action == 3 and np.any(br_mask * cur_state.reshape(4, 4)):
        return 0.8
    return -.5
    
def get_next_state():
    possible_states = [
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]])
        ]
    return random.sample(possible_states, 1)[0]

def test_model(model):
    test_states = [
        np.array([[1, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1]]),
        np.array([[0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1]])
        ]
    
    true_labels = [0, 1, 2, 3]
    correct = 0.0
    for state, label in zip(test_states, true_labels):
        if model.act(state.reshape(1, 16)) == label:
            correct = correct + 1.0
    return correct / len(test_states)


if __name__ == "__main__":
    print_examples = False
    gamma   = 0.9
    epsilon = .98

    trials  = 10
    trial_len = 200

    dqn_agent = DQN()
    steps = []
    for trial in range(trials):
        cur_state = get_next_state()
        cur_state = cur_state.reshape(1, 16)
        
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state = get_next_state()
            reward = get_reward(cur_state, action)
            done = False

            if print_examples and step < 5:
                print('trial', trial, 'step', step)
                print('current_state')
                print(cur_state.reshape(4, 4))
                print('action')
                print(action)
                print('reward')
                print(reward)

            new_state = new_state.reshape(1,16)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                break

        print(f"Finished trial {trial}")
        test_accuracy = test_model(dqn_agent)
        print(f"Accuracy: {test_accuracy}")
