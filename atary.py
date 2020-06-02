import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import gym
import random
import numpy as np
from collections import deque
import os.path
from datetime import datetime

from gym_wrappers import wrap_deepmind
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense

class DQNAgent():
    def __init__(self, game_name="Breakout", total_episodes=None, render=True, clip_reward = True):
        #env settings
        self.env_name = game_name + "Deterministic-v4"  
        self.total_episodes_limit = total_episodes
        self.render = render
        self.clip_reward = clip_reward
        self.target_model_update = 40000
        self.model_save_frequency = 10000

        #using openai wrapper to pre-process the input
        env = gym.make(self.env_name)
        self.env = wrap_deepmind(env, True, False, True)
        self.input_shape = (84, 84, 4)
        self.action_size = self.env.action_space.n

        #replay memory and sample
        self.memory = deque(maxlen=200000)
        self.mem_sample = 32
        self.observation_size = 50000
        self.train_frequency = 4

        #learning rate
        self.alpha = 0.00025

        #exploration
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_steps = 500000
        self.epsilon_reduction = (self.epsilon - self.epsilon_min)/self.epsilon_steps

        self.gamma = 0.99

        #init models
        self.train_model = self.create_atari_model()
        self.target_model = self.create_atari_model()
        self.target_model.set_weights(self.train_model.get_weights())

    def create_atari_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=8, strides=4,
                              padding="same",
                              activation="relu",
                              input_shape=self.input_shape))
        self.model.add(Conv2D(64, kernel_size=4, strides=2,
                              padding="same",
                              activation="relu",
                              input_shape=self.input_shape))
        self.model.add(Conv2D(64, kernel_size=3, strides=1,
                              padding="same",
                              activation="relu",
                              input_shape=self.input_shape))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(self.action_size))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.alpha))
        self.model.summary()
        return self.model
        

    def save_model(self):
        self.train_model.save(self.env_name)

    def load_model(self):
        self.target_model.load_weights(self.env_name)
        self.train_model.load_weights(self.env_name)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon or len(self.memory) < self.observation_size:
            return random.randrange(self.action_size)
        q_values = self.train_model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])

    def memorize(self, current_state, action, reward, next_state, terminal):
        self.memory.append((current_state, action, reward, next_state, terminal))

    def train_memory(self, total_step):
        if len(self.memory) < self.observation_size:
            return

        if total_step % self.train_frequency == 0:
            minibatch = np.asarray(random.sample(self.memory, self.mem_sample))
            if len(minibatch) < self.mem_sample:
                return

            current_states = []
            q_values = []
            max_q_values = []

            for state, action, reward, next_state, terminal in minibatch:
                current_state = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
                current_states.append(current_state)
                next_state = np.expand_dims(np.asarray(next_state).astype(np.float64), axis=0)
                next_state_prediction = self.target_model.predict(next_state)
                
                next_q_value = np.max(next_state_prediction)
                
                q = list(self.train_model.predict(current_state)[0])
                if terminal:
                    q[action] = reward
                else:
                    q[action] = reward + self.gamma * next_q_value
                q_values.append(q)
                max_q_values.append(np.max(q))

            fit = self.train_model.fit(np.asarray(current_states).squeeze(),
                                np.asarray(q_values).squeeze(),
                                batch_size=self.mem_sample,
                                verbose=0)
            #loss = fit.history["loss"][0]
            #accuracy = fit.history["acc"][0]
            
            
        #epsilon decay
        self.epsilon -= self.epsilon_reduction
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if total_step % self.model_save_frequency == 0:
            self.save_model()

        if total_step % self.target_model_update == 0:
            self.target_model.set_weights(self.train_model.get_weights())
            print("updating model, epsilon: {}, total steps: {}".format(self.epsilon, total_step))

    def train(self):
        episode = 0
        total_steps = 0

        while True:
            if self.total_episodes_limit is not None and episode >= self.total_episodes_limit:
                print("Episodes completed: " + str(self.total_episodes_limit))
                exit(0)
            
            episode += 1
            current_state = self.env.reset()
            step = 0
            score = 0
            print("Episode: {}, total steps: {}".format(episode, total_steps))

            while True:
                total_steps += 1
                step += 1

                if self.render:
                    self.env.render()

                action = self.choose_action(current_state)
                next_state, reward, terminal, info = self.env.step(action)

                if self.clip_reward:
                    np.sign(reward)

                score += reward
                self.memorize(current_state, action, reward, next_state, terminal)
                current_state = next_state

                self.train_memory(total_steps)

                #game over
                if terminal:
                    break

    def play(self, filename, trials=400, render=True):
        epochs = []
        successes = []
        total_rewards = []
        self.load_model(filename)
        try:
            for episode in range(trials):
                current_state = self.env.reset()

                terminal=False
                total_reward = 0
                while not terminal:
                    if render:
                        self.env.render()
                    action = np.argmax(self.train_model.predict(np.expand_dims(np.asarray(current_state).astype(np.float64), axis=0), batch_size=1)[0])

                    next_state, reward, terminal, info = env.step(action)

                    total_reward += reward
                    state = next_state
               
                print("Episode #{} total reward {}".format(episode + 1, total_reward))
                total_rewards.append(total_reward)
        finally:
            self.env.close()

        return total_rewards 

dqn=DQNAgent()
dqn.train()