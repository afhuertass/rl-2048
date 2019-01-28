import gym 
from gym import error, spaces, utils 

from gym.utils import seeding 
from random import *

import numpy as np
from scipy.ndimage.filters import convolve1d as conv
import logging
logger = logging.getLogger(__name__)


class Game2048( gym.Env ):

	metadata = {'render.modes': ['human']}
	
	def __init__(self , n = 4 ):

		self._score = 0 
		self.N = n 
		self.score =  0
		self.done = False 
		# 4 posibles acciones 
		self.action_space = spaces.Discrete( 4 )

		self.observation_space = spaces.Discrete( n*n )

		self.step_reward = 0

		self.matrix = np.zeros( ( n , n ))
		self.actions = [ "up" , "down" , "left" , "right" ]

		self.action2vector = {}
		self.action2vector["up"]  = np.array( [1 , 0 , 0 , 0])
		self.action2vector["down"]  = np.array( [0 , 1 , 0 , 0])
		self.action2vector["left"]  = np.array( [0 , 0 , 1 , 0])
		self.action2vector["right"]  = np.array( [0 , 0 , 0 , 1 ])

		d = 0.5 / 7 
		self.Ps = [ 0.5 ,  0.25 , 0.25 ]
		print( sum( self.Ps))
		self.Numbers = [  0 , 2 , 4   ]
		self.n_zero = 0 
		return

	def _seed(self):

		pass

	def getaction2vector( self , action):

		return self.action2vector[action] 
		 
	def add_two(self):
		# add new tile with a number two
		#print( self.matrix )
		a = np.array( self.matrix ) 
		indxs = np.argwhere( a == 0 )
		#print( indxs )
		if len( indxs) == 0 : 
			return None 
		i = np.random.choice( np.arange( len(indxs)) , 1  ) 
		ix = indxs[i]

		#print( ix )
		self.matrix [ ix[0][0] ] [ ix[0][1] ]  = 2 
		#print( self.matrix )

	def step( self , action ):
		# plan de rewards 
		# Si el movimiento no genera ninguna reaccion ( delta = -2 ) la recompensa sera de -2
		# si el movimiento genera alguna reaccion, la recompensa sera el maximo en la pantalla mas la diferencia en el score
		reward = 0 
		score_prev = self.score
		delta , m  = self._take_action( action )
		diff = self.score - score_prev 
		done = False
		reward = 0 
		if self.status == "lose":
			done = True
			reward = -1
			#print("LOSE")
		if self.status == "win":
			reward = 1
			done = True
			print( "WIN!")


		rep = self.matrix2input( )

		#return np.array( self.matrix ).reshape( 4,4 ) , reward , done , {}
		return  rep , reward , done , {}

	def _get_status2( self ):

		a = np.array( self.matrix).reshape( 4,4 )

		n = 1
		k = np.ones(n,dtype=int)
		v = (conv((a[1:]==a[:-1]).astype(int),k,axis=0,mode='constant')>=n).any()
		h = (conv((a[:,1:]==a[:,:-1]).astype(int) ,k,axis=1,mode='constant')>=n).any()

		if a.max() >=  256 :
			return "win"

		if  v | h:
			return "not_over"


		nzeros = np.count_nonzero(  a  )
		if nzeros >= 16 :
			return "lose"



	def matrix2input( self ):

		valids = [ 0 , 2 , 4 , 8 , 16 , 32 , 64 , 128 , 256 , 512, 1024 , 2048 ]
		a = np.array( self.matrix )
		inputs = np.zeros( (  4 , 4 , len(valids)   )  )
		for i , v  in enumerate( valids): 
			rep = (a == v).astype( int )
			inputs[ : , : , i  ]  = rep 


		return inputs 



	def _get_status(self):
		# check for winning condition
		for i in range( self.N ):
			for j in range( self.N ):
				if self.matrix[i][j] == 2048:
					return "win"
		# check for any zeros

		for i in range( self.N ):
			for j in range( self.N ):
				if self.matrix[i][j] == 0 :
					return "not_over"
		# check for adjecent equal numbers, a move can me made
		for i in range( self.N - 1 ):
			for j  in range( self.N - 1 ):
				if self.matrix[i][j] == self.matrix[i+1][j] or self.matrix[i][j+1] == self.matrix[i][j]:
					return "not_over"


		for k in range( self.N - 1) :
			if self.matrix[ self.N - 1 ][k] == self.matrix[self.N -1 ][k+1]:
				return "not_over"

		for j in range( self.N -1) :
			if self.matrix[j][self.N - 1 ] == self.matrix[j+1][self.N-1]:
				return "not_over"

		return "lose"

	def _reverse(self , M ):

		new = []
		for i in range( self.N ):
			new.append( [] )
			for j in range( self.N ):
				new[i].append( M[i][self.N -1 - j]  )

		return new 

	def _cover_up( self , M = None ):
		new = [ [0]*self.N for i in range(self.N)]

		done = False

		for i in range(self.N):
			count = 0 
			for j in range(self.N):
				if M[i][j] != 0:
					new[i][count] = M[i][j] 
					if j!= count:
						done = True
					count +=1

		return  new , done 

	def merge( self , M ):

		done = False
		count = 1 
		for i in range( self.N ):
			for j in range(self.N-1):
				if M[i][j] == M[i][j+1] and M[i][j] != 0 :
					M[i][j] *= 2

					M[i][j+1] = 0
					self.score += M[i][j]
					count += 1 
					done = True 
		return M , done , count   

	def _transpose(self , M ):

		new = []
		for i in range( self.N ):
			new.append([])
			for j in range( self.N ):
				new[i].append( M[j][i])

		return new

	def up(self):

		board = self._transpose( self.matrix)
		board , done = self._cover_up( board )
		tmp = self.merge( board )
		game = tmp[0]
		done = done or tmp[1]

		game = self._cover_up(game )[0]
		game = self._transpose( game)

		done = done or tmp[1]
		self.matrix = game
		self.done = done

		return tmp[2]

	def down(self):

		board  = self._reverse( self._transpose( self.matrix ))
		game , done = self._cover_up( board )
		tmp = self.merge( game )
		game = tmp[0]

		game = self._cover_up( game )[0]
		game = self._transpose( self._reverse( game ))
		done = done or tmp[1]
		self.matrix = game
		self.done = done

		return tmp[2]

	def left( self ): 

		game , done = self._cover_up( self.matrix )
		tmp = self.merge( game )
		game = tmp[0]

		game = self._cover_up(game)[0]
		done = done or tmp[1]
		self.matrix = game
		self.done = done

		return tmp[2]

	def right(self):

		game  = self._reverse( self.matrix ) 
		game , done = self._cover_up( game )
		tmp = self.merge( game )
		game = tmp[0]
		done = done or tmp[1]
		game = self._cover_up(game)[0]
		game = self._reverse( game )

		self.matrix = game
		self.done = done 
		return tmp[2]

	def _take_action( self , action ):

		delta = 0
		m = 1 
		if action =="up":
			m = self.up()
		elif action =="down":
			m = self.down()
		elif action =="left":
			m = self.left()
		elif action =="right":
			m = self.right()

		#self.add_two()
		self.status = self._get_status2()
		if not self.done:

			delta = -2

		if self.done:
			self.add_two()

			self.done = False

			#status = self._get_status2()
			"""
			if status == "win":
				print("win")
			if status == "lose":
				print("lose")
			"""

		return delta , m

	def reset(self) :
		self.score = 0
		self.done = False
		self.step_reward = 0
		self.n_zero = 0 
		#self.matrix = np.zeros( (self.N, self.N) )
		self.status = "not_over"
		self.matrix =  np.random.choice( a = self.Numbers ,  size = ( 4,4) , p = self.Ps )
		rep = self.matrix2input()
		#return np.array( self.matrix ).reshape(self.N , self.N )
		print( rep.shape )
		return rep 

	def random_action(self):
		ii = randint( 0 , self.N - 1  ) 

		return self.actions[ ii ]

	def sample_action( self , P ):
		#print( P )
		return np.random.choice( self.actions , p = P )

	def _observe(self):
		pass
	def _reward(self):

		pass 
	def _render(self):
		pass

	def _render(self , mode = "human" , close = False ):
		return