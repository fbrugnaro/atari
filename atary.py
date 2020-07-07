import tensorflow as tf
import gym
import random
import numpy as np

from gym_wrappers import wrap_deepmind, make_atari
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input

class DDQNAgent():
    def __init__(self, load_model=True, game_name="Breakout", testing=False, total_episodes=None, render=False):
        #env settings
        self.seed = 3
        self.game_name = game_name
        self.env_name = game_name + "NoFrameskip-v4"
        if testing:
            self.env_name = game_name + "-v4"

        self.total_episodes_limit = total_episodes
        self.render = render
        self.target_model_update = 10000
        self.save_counter = 0

        #using openai wrapper to pre-process the input
        if testing:
            env = gym.make(self.env_name)
        else:
            env = make_atari(self.env_name)

        self.env = wrap_deepmind(env, True, True, True, True)
        self.env.seed(self.seed)
        self.action_size = self.env.action_space.n

        #replay memory and sample
        self.action_history = []
        self.state_history = []
        self.next_state_history = []
        self.rewards_history = []
        self.terminal_history = []
        self.max_memory_len = 900000
        self.mem_sample = 32
        self.observation_size = 50000
        self.train_frequency = 4

        #exploration
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_steps = 1000000
        self.epsilon_reduction = (self.epsilon - self.epsilon_min)/self.epsilon_steps

        self.gamma = 0.99

        #init models
        self.train_model = self.create_atari_model()
        self.target_model = self.create_atari_model()
        self.target_model.set_weights(self.train_model.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        if load_model or testing:
            self.reload_model()
            self.epsilon = 0.1
 

    def create_atari_model(self):
        inputs = Input(shape=(84, 84, 4,))

        layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)
        layer4 = Flatten()(layer3)
        layer5 = Dense(512, activation="relu")(layer4)
        action = Dense(self.action_size, activation="linear")(layer5)

        model = Model(inputs=inputs, outputs=action)
        model.summary()

        return model
        

    def save_model(self):
        print ("Saving model epsilon: {}".format(self.epsilon))
        self.target_model.save("Atari/{}-{}".format(self.game_name, self.save_counter))
        self.save_counter += 1
        if self.save_counter == 3:
            self.save_counter = 0

    def reload_model(self):
        print ("Loading model")
        self.target_model = load_model("Atari/{}".format(self.game_name))
        self.train_model = load_model("Atari/{}".format(self.game_name))

    def choose_action(self, state, total_steps):
        if np.random.rand() < self.epsilon or total_steps < self.observation_size:
            return np.random.choice(self.action_size)
            
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.train_model(state_tensor, training=False)

        return tf.argmax(action_probs[0]).numpy() 

    def choose_best_action(self, state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.train_model(state_tensor, training=False)

        return tf.argmax(action_probs[0]).numpy()

    def memorize(self, current_state, action, reward, next_state, terminal):
        self.state_history.append(current_state)
        self.action_history.append(action)
        self.rewards_history.append(reward)
        self.next_state_history.append(next_state)
        self.terminal_history.append(terminal)

    def train_memory(self, total_step):
        if total_step % self.train_frequency == 0 and len(self.state_history) > self.mem_sample:
            #sample random batch
            indices = np.random.choice(range(len(self.state_history)), size=self.mem_sample)
            
            state_sample = np.array([self.state_history[i] for i in indices])
            next_state_sample = np.array([self.next_state_history[i] for i in indices])
            rewards_sample = [self.rewards_history[i] for i in indices]
            action_sample = [self.action_history[i] for i in indices]
            terminal_sample = tf.convert_to_tensor([float(self.terminal_history[i]) for i in indices])
            
            #predict using the target model
            future_rewards = self.target_model.predict(next_state_sample)
            updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
            
            #if terminal set the last value to -1
            updated_q_values = updated_q_values * (1 - terminal_sample) - terminal_sample
            
            masks = tf.one_hot(action_sample, self.action_size)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.train_model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                # Clip the deltas using huber loss for stability
                huber = keras.losses.Huber()
                loss = huber(updated_q_values, q_action)

            #backpropagation
            grads = tape.gradient(loss, self.train_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.train_model.trainable_variables))
            
        
    def train(self):
        episode = 0
        total_steps = 0

        while True:
            if self.total_episodes_limit is not None and episode >= self.total_episodes_limit:
                print("Episodes completed: " + str(self.total_episodes_limit))
                exit(0)
            
            episode += 1
            current_state = np.array(self.env.reset())
            step = 0
            episode_reward = 0

            while True:
                total_steps += 1
                step += 1

                if self.render:
                    self.env.render()

                action = self.choose_action(current_state, total_steps)

                #epsilon decay
                self.epsilon -= self.epsilon_reduction
                self.epsilon = max(self.epsilon_min, self.epsilon)

                next_state, reward, terminal, info = self.env.step(action)
                next_state = np.array(next_state)

                episode_reward += reward
                
                #save in the replay buffer
                self.memorize(current_state, action, reward, next_state, terminal)
                current_state = next_state

                #train the model
                self.train_memory(total_steps)                    

                #update & save target model
                if total_steps % self.target_model_update == 0:
                    self.target_model.set_weights(self.train_model.get_weights())
                    print("updating model, epsilon: {}, total steps: {}".format(self.epsilon, total_steps))
                    self.save_model()

                #limit memory len
                if len(self.state_history) > self.max_memory_len:
                    del self.rewards_history[:1]
                    del self.state_history[:1]
                    del self.next_state_history[:1]
                    del self.action_history[:1]
                    del self.terminal_history[:1]


                #episode is over
                if terminal:
                    print("Episode: {}, total steps: {}, score: {}, epsilon: {}".format(episode, total_steps, episode_reward, self.epsilon))
                    break

    def test(self, trials=500, render=True):
        total_scores = []
        game_score = 0
        
        for episode in range(trials):
            current_state = self.env.reset()
            score = 0

            while True:
                if render:
                    self.env.render()
                action = self.choose_best_action(current_state)

                next_state, reward, terminal, info = self.env.step(action)

                score += reward
                current_state = next_state

                if terminal:
                    print("Episode number: {}, this episode score: {}".format(episode + 1, score))
                    game_score += score
                    total_scores.append(score)
                    break
            
            if info['ale.lives'] == 0:
                print("Game Over, total score: {}".format(game_score))
                game_score = 0
        
        
        self.env.close()

        return total_scores 

dqn=DDQNAgent(load_model=False, total_episodes=2000000)
dqn.train()

dqn=DDQNAgent(True, testing=True)
scores = dqn.test()
hiscore = 0
for s in scores:
    if s > hiscore:
        hiscore = s
print("hiscore: {}".format(hiscore))