import gym
import gym_2048
from models import RandomAgent
from models import MasterAgent

#print(envs.registry.all())
#aaa = gym.make("game2048-v0")
#randAgent = RandomAgent("game2048-v0" , 50 )

#randAgent.run()

master = MasterAgent()
master.train()
#master.play( "../data/shared2/conv-new-reward-game2048-v0-3276-ep-46243.h5"  , "../data/shared2/game/")