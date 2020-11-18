from collections import deque
import os
import random

from IPython.core.display import clear_output
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory, MemoryBufferPER
import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_00
WARM_STEPS = 1_000
MAX_STEPS = 10_000_0
EVALUATE_FREQ = 100_0

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
#os.mkdir(SAVE_PREFIX)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
)

prioritized = True
if prioritized:
    memory = MemoryBufferPER(STACK_SIZE + 1, MEM_SIZE, device)
else:
    memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True
episode_reward = 0
loss = 0
losses = []
all_rewards = []
progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)
        all_rewards.append(episode_reward)
        episode_reward = 0

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    episode_reward += reward
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

    if step % POLICY_UPDATE == 0 and training:
        loss = agent.learn(memory, BATCH_SIZE)
        losses.append(loss)

    if step % TARGET_UPDATE == 0:
        agent.sync()

    #if step % EVALUATE_FREQ == 0:
        # avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        # with open("rewards.txt", "a") as fp:
        #     fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        # if RENDER:
        #     prefix = f"eval_{step//EVALUATE_FREQ:03d}"
        #     os.mkdir(prefix)
        #     for ind, frame in enumerate(frames):
        #         with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
        #             frame.save(fp, format="png")
        # agent.save(os.path.join(
        #     SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        #done = True

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_training(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[:])))
    plt.plot(moving_average(rewards,20))
    plt.subplot(132)
    plt.title('loss, average on 100 stpes')
    plt.plot(moving_average(losses, 100),linewidth=0.2)
    plt.show()

plot_training(step, all_rewards, losses)
