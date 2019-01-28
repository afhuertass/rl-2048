
import tensorflow as tf
#from tensorflow.keras import layers 
#from tensorflow.python import keras
from tensorflow.python.keras import layers
from queue import Queue
import gym 
import threading
import multiprocessing
import os
import numpy as np
import tensorflow.contrib.eager as tfe
#tfe.

import matplotlib.pyplot as plt
import scipy.signal

import seaborn as sns
tfe.enable_eager_execution()


def record(  global_ep_reward , result_queue , episode_reward):

	if global_ep_reward == 0:
		global_ep_reward = episode_reward
	else:
		global_ep_reward = global_ep_reward*0.99 + episode_reward*0.01


	result_queue.put( global_ep_reward )
	return global_ep_reward



class ActorCriticModel( tf.keras.Model ):


	def __init__(self , state_size  ,  nlayers = 4  ,action_size = 4  ):
		super(  ActorCriticModel , self).__init__()

		self.state_size = state_size
		self.action_size = action_size 
		self.nhidden = 1024

		self.shared = []
		#input_layer = layers.Dense( self.nhidden , activation = "relu" , input_shape = (4,4 , ) , kernel_initializer = "lecun_normal" )
		#input_layer = layers.Reshape( (4,4 , 1 ))
		#self.shared.append ( input_layer )
		for i in range( nlayers ):

			#ayer = layers.Dense( self.nhidden , activation = "relu" , kernel_initializer = "lecun_normal" )
			#layer = layers.TimeDistributed ( layers.Conv2D( 32 , kernel_size = (1,1) , strides = (1,1) , padding = "VALID" ) ) 

			layer = layers.Conv2D( 16 ,  kernel_size = ( 2, 2 ) , strides = (1,1 ) , activation = "relu" , kernel_initializer = "lecun_normal" , padding = "same") 
			self.shared.append( layer )
			layer = layers.Conv2DTranspose( 16 ,  kernel_size = ( 2, 2 ) , strides = (1,1 ) , activation = "relu" , kernel_initializer = "lecun_normal" , padding = "same") 
			self.shared.append( layer )
			
			#self.layers_policy.append( do )

		self.shared.append( layers.TimeDistributed( layers.Flatten() )  )
		self.shared.append( layers.LSTM( self.nhidden  ) ) 
		self.shared.append( layers.Flatten() )
		#self.shared.append( layers.Dropout( 0.2 )  )

		self.logits = layers.Dense( 4  , activation = "linear" , kernel_initializer = "lecun_normal")
		self.values = layers.Dense( 1 , activation = "linear" , kernel_initializer = "lecun_normal" )


	def call( self , inputs ):

		"""
		x = inputs 
		x2 = inputs
		for layer in self.layers_policy:

			x = layer(x )

		logits = x 

		for layer in self.layers_values:
			x2 = layer(x2)

		values = x2

		"""
		x = inputs 
		for layer in self.shared:
			#print(layer)
			x = layer( x)

		logits = self.logits(x)
		values = self.values(x)

		return logits , values 


class RandomAgent:

	def __init__(self , env_name , max_eps):

		self.env = gym.make( env_name )
		self.max_eps = max_eps 


	def run(self):

		reward_avg = 0 
		for episode in range(self.max_eps):

			self.env.reset()

			steps = 0
			done = False
			reward_sum = 0.0 
			while not done:

				_ , reward , done , _ = self.env.step( self.env.random_action() )
				steps += 1
				reward_sum += reward

			reward_avg += reward_sum

		reward_avg = reward_avg/(float(self.max_eps))
		print( "Avg reward across {} episodes: {}".format(self.max_eps , reward_avg) )

		

class Memory:


	def __init__(self):

		self.states = []
		self.actions = []

		self.actions = []
		self.rewards = []


	def store(self , state, action , reward):

		self.states.append(state)
		self.actions.append( action )
		self.rewards.append( reward )

	def clear(self):

		self.states = []
		self.actions = []
		self.rewards = []

class MasterAgent():

	def __init__(self, save_dir = "../data/model"):
		self.game = "game2048-v0"
		self.save_dir  = save_dir 

		if not os.path.exists( save_dir ):
			os.makedirs( save_dir )

		env = gym.make( self.game )
		self.state_size = env.observation_space.n 
		self.action_size = env.action_space.n 

		self.opt = tf.train.AdamOptimizer( 1e-6 , use_locking = True )
		self.global_model = ActorCriticModel( self.state_size , self.action_size )
		self.global_model( tf.convert_to_tensor( np.random.random( ( 1 , 4 ,4 , 12   ))  , dtype = tf.float32) )



	def train( self ):

		result_queue = Queue()

		workers = [ Worker( self.state_size ,
			self.action_size , 
			self.global_model , 
			self.opt , 
			result_queue , 
			i , game = self.game , save_dir = "../data/shared2" , max_eps = 1000 ) for i in range( multiprocessing.cpu_count())		]


		#workers[0].run()
		for i , worker in enumerate(workers):

		#	print( "starting worker {}".format(i) )

			worker.start()
			#break
		moving_average_rewards = [ ]
		while True:

			reward = result_queue.get()
			if reward is not None:
				moving_average_rewards.append( reward )
			else:
				break 

		[ w.join() for w in workers ]

		plt.plot(moving_average_rewards)
		plt.ylabel('Moving average ep reward')
		plt.xlabel('Step')
		plt.savefig(os.path.join(self.save_dir,'2048 Moving Average.png') )
		plt.show()


			
	def play( self , weights_path ,  output_dir  ,tries = 10   ):

		self.global_model.load_weights( weights_path )
		self.env = gym.make( self.game )

		self.env.reset()
		current_state = self.env.matrix
		done = False
		counter = 0 
		while not done:

			print( "Prev Stat")
			print( current_state )
			transformed_state = self.env.matrix2input()
			print( transformed_state.shape )
			logits , _  = self.global_model( tf.convert_to_tensor( transformed_state.reshape( 1 ,  4, 4, 12 ) , dtype = tf.float32  ))

			probs = tf.nn.softmax( logits )
				#action = np.random.choice( self.local_model.actions , p = probs.numpy()[0] )
			action = self.env.sample_action( probs.numpy()[0] )
			new_state  , reward , done , _ = self.env.step( action )
			ax = sns.heatmap( self.env.matrix  , vmin = 0 , vmax = 256 , annot = True )
			print( "Action - new State")
			print( action )
			print( new_state )
			print( reward )
			print( self.env.score )
			fig = ax.get_figure()
			current_state = new_state
			filename = "game_{}.png".format( counter )
			counter += 1 
			fig.savefig( output_dir + filename)
			plt.clf()

		print( self.env.score )



class Worker( threading.Thread ):


	global_episode = 0 
	best_score = -100
	best_reward = 0 
	save_lock = threading.Lock()
	global_moving_average_reward = 0 
	def __init__(self , state_size  , action_size  ,global_model , opt , result_queue , idx , game  , save_dir  , max_eps ):

		super( Worker , self).__init__()
		self.state_size = state_size 
		self.action_size = action_size 
		self.result_queue = result_queue 
		self.global_model = global_model 
		self.opt = opt 
		self.local_model = ActorCriticModel( self.state_size , self.action_size )

		self.worker_idx = idx  
		self.game_name = game
		self.env = gym.make(  self.game_name )
		self.save_dir = save_dir
		self.ep_loss = 0.0 
		self.max_eps = max_eps

		self.gamma = 0.99
		pass 


	def run(self):

		total_step = 1 
		mem = Memory()

		n_episodes = 0 

		scores = []
		rewards_full = []
		losses = []
		steps_death = [ ]

		best_scores = []
		while Worker.global_episode  < self.max_eps :
			print( self.max_eps )
			print( Worker.global_episode )
			print( "*" *10)
			current_state = self.env.reset()
			
			mem.clear()
			ep_reward = 0 
			ep_steps = 0 
			self.ep_loss = 0

			local_steps = 0 
			done = False 

			avg_rewards = []
			print( "TERMINATOR ")
			while not done:
				logits , _  = self.local_model( tf.convert_to_tensor( current_state.reshape( 1 , 4 ,4 , 12 ) , dtype = tf.float32  ))

				probs = tf.nn.softmax( logits )
				#action = np.random.choice( self.local_model.actions , p = probs.numpy()[0] )
				action = self.env.sample_action( probs.numpy()[0] )
				#print(action)
				new_state  , reward , done , _ = self.env.step( action )

				mem.store( current_state , action , reward )

				ep_steps += 1
				local_steps += 1
				ep_reward += reward
				avg_rewards.append( reward )
				if done or local_steps % 25 == 0 : 
					# si ya acabo el

					#done = True 

					with tf.GradientTape() as tape:
						total_loss = self.compute_loss( done , new_state , mem , self.gamma)

					self.ep_loss += total_loss

					grads = tape.gradient( total_loss , self.local_model.trainable_weights )
						#grads , _  =  tf.clip_by_global_norm( grads , 20.0)
					self.opt.apply_gradients( zip( grads , self.global_model.trainable_weights ))

					self.local_model.set_weights( self.global_model.get_weights() )

					mem.clear()
					time_count = 0 
					#rewards_full.append( ep_reward )

				if done:
					
					Worker.global_moving_average_reward = record( 
						Worker.global_moving_average_reward , self.result_queue , ep_reward
						)
					if ep_reward >= 1.0:

						with Worker.save_lock:

							print( "saving best reward model, solve the challenge" )
							self.global_model.save_weights(
								self.save_dir + "/conv-compete-{}-ep-{}.h5".format( self.env.score , Worker.global_episode )
								)
							Worker.best_reward = ep_reward

					if self.env.score > Worker.best_score:

						with Worker.save_lock:

							print( "saving model bestscore episode reward, env_score , max : {} , {} , {}".format( ep_reward, self.env.score , np.max(self.env.matrix )) )
							self.global_model.save_weights(
								self.save_dir + "/conv-bestscore-{}-ep-{}.h5".format( self.env.score , Worker.global_episode )
								)
							Worker.best_score = self.env.score 



					print( "Worker: " , self.worker_idx )
					print( "Reward" , ep_reward )
					print( "Max" , np.array(self.env.matrix).max()  )
					print( "Score: " , self.env.score )
					print( "Global Counter" , Worker.global_episode )
					print( "Local steps" , local_steps  )

					

					n_episodes += 1
					Worker.global_episode += 1

				current_state = new_state 



			scores.append( self.env.score )


		
		self.result_queue.put(None)


	def discount( self , x, gamma):
		return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

	def compute_loss( self ,   done , new_state , mem , gamma ):


		#print("State")
		#print( new_state)
		#print( "Rewards")
		#print( mem.rewards )
		#print( "Sum rewards")
		#print( sum( mem.rewards ))
		reward_sum = 0 
		reward_sum = self.local_model(tf.convert_to_tensor(new_state[None, :],
				dtype=tf.float32))[-1].numpy()[0]


		reward_sum = reward_sum[0]

		print( "REWARD SUM" )
		print( reward_sum)

		discounted_rewards = []
		for reward in mem.rewards[::-1]:
			reward_sum = reward_sum + gamma*reward 
			discounted_rewards.append( reward_sum )

		discounted_rewards.reverse()
		#print( "Memory rewards")

		#print( mem.rewards )
		#discounted_rewards =  self.discount( mem.rewards , gamma )
		#print( "Discount rewards ")

		#print( discounted_rewards )
		L = len( mem.states )
		mem_states = np.array(  mem.states ).reshape( L ,  4 , 4 , 12  )
		#print( mem_states.shape )
		#print( type( mem.states[0] )  )
		#for act , mem_state , rew in zip( mem.actions , mem_states , mem.rewards ):
		#	print( act )
		#	print( mem_state )
		#	print( rew )

		logits , val = self.local_model( 
				tf.convert_to_tensor( mem_states 
				, dtype = tf.float32 )
			)

		advantage = tf.convert_to_tensor(
			np.array( discounted_rewards )[:, None  ] ,
			dtype = tf.float32 ) - val
		labels_policy = np.vstack( [ self.env.getaction2vector(act) for act in  mem.actions ] )

		policy = tf.nn.softmax( logits )
		log_prob = tf.log(  tf.reduce_sum ( policy*labels_policy , reduction_indices = 1   ) )
		policy_loss = - log_prob*( advantage )
		value_loss = tf.reduce_mean( tf.square( advantage ))
		total_loss = tf.reduce_mean(  policy_loss + ( 0.5*value_loss ) ) 
		"""
		policy = tf.nn.softmax( logits )
		labels_policy = np.vstack( [ self.env.getaction2vector(act) for act in  mem.actions ] )

		value_loss =   tf.square( advantage ) 
		entropy = tf.reduce_sum( tf.log( policy + 1e-20  )* policy , axis = 1    )

		policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2( labels = labels_policy , logits = logits  )
		policy_loss *= tf.stop_gradient( advantage ) 
		policy_loss -= 0.01*entropy 

		#print("POLICY LOSS")
		#print( value_loss )
		total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
		"""

		return total_loss 