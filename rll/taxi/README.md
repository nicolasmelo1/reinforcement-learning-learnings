# Taxi problem

This is where i started my journey into reinforcement learning.

I'm using the [Taxi problem](https://gymnasium.farama.org/environments/toy_text/taxi/) from gymnasium.

## Learnings

- Q-Tables. I'm using a Q-table to store the values of the states and map to each action.

Q-Table is pretty much a matrix where the rows are the states and the columns are the actions. During the training phase we pretty much update the values of this table until we have the desired best scenario

- Epsilon. Episilon is an hyperparameter that is used to control the exploration of the environment. It's a value between 0 and 1. The higher the value, the more the AI will explore other actions, the lower the value, it'll follow what it knows. We use a decay episilon so we start with a high Epsilon, so we explore a lot the environment, and then decay the exploration.

- Alpha. Alpha is the learning rate. It's a value between 0 and 1. The higher the value, the faster the AI will learn. We used a hyperparameter tuner. In other words, we put it in an array so we test multiple hyper parameters on a single run. We found the best value is 1.

- Gamma. Gamma is the discount factor. It's a value between 0 and 1. The higher the value the more AI will want to maximize the most immediate reward. The lower the value, the AI will want to maximize the immediate reward. It wants to act the way it will receive the max reward immeadiately. If it's a higher value it'll try to maximize the total sum of rewards of all actions

- Agent. The agent is the AI that will be learning. It's a class that has the Q-table, the Epsilon and the Alpha. This is where the magic happens:

```python
def update(self, state, action, reward, next_state):
  old_value = self.q_table[state][action]
  self.q_table[state][action] = self.q_table[state][action] + self.alpha * (
      reward + self.gamma * np.max(self.q_table[next_state]) - old_value
  )
```

You can see that the agent has a method called `update` that takes the state, the action, the reward and the next state. It then calculates the old value of the state and action, and then updates the Q-table with the new value.

The update method is called every time the agent takes an action.

## Running the project

To run the project, you can use the following command:

```bash
uv main.py taxi train
```

This will train the model and you can see the agent learning the environment.

To run the trained model, you can use the following command:

```bash
uv main.py taxi run model_alpha_1_gamma_0-9.npy
```

## Homeworks

In the first challenge, I dare you to update the train()function src/loops.py to accept an episode-dependent epsilon.

In the second challenge, I want you to upgrade your Python skills and implement paralleling processing to speed up hyper-parameter experimentation.

Both challenges were implemented. The first, is on utils.py
The second was done using python's own multiprocessing library. We use multiprocessing so we can run multiple processes at the same time. Multithreading is not Ideal for this. We can stop multiple processes, we also delegate to the OS to control the multiple processes, we don't have such thing using multithreading
