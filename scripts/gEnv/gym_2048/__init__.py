
from gym.envs.registration import register



register( 
	id = "game2048-v0" , 
	entry_point = "gym_2048.envs:Game2048"

	)