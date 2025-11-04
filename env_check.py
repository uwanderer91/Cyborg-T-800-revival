from env import *
import matplotlib.pyplot as plt

env = DiepEnv()
state = env.reset()

#for i in range(0, 19):
#    state, reward, done, _ = env.step(4)

plt.figure(figsize=(6, 6))
plt.imshow(state.squeeze(), cmap='gray')
plt.show()