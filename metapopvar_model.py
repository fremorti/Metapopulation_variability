'''
Created on Feb 27, 2018
Author: fremorti
'''

import random as rnd
import numpy as np
import math as math
import time
start = time.clock()
default_path = os.getcwd()

class Individual:
    '''Class that regulates individuals and their properties'''
    def __init__(self,
                 x,
                 y
                 ):
        '''Initialization'''
        self.x = x                  #location (x, y)
        self.y = y
		self.dispprop = 0.2
        self.age = 0
		self.fec = 2
        
        
    def move(self, max_x, max_y):
        '''an individual moves 
        '''
        dx, dy = rnd.sample((-1, 0, 1)), rnd.sample((-1, 0, 1))
		while not(0<=self.y+dy<=max_y and 0<=self.x+dx<=max_x and dx,dy != 0, 0):
			dx, dy = rnd.sample((-1, 0, 1)), rnd.sample((-1, 0, 1))
		self.x, self.y += dx, dy
        
   
class Metapopulation:
    '''Contains the whole population, regulates daily affairs'''
    def __init__(self,
                 max_x,
                 max_y,
                 r,
				 K,):
        '''Initialization'''           
        self.max_x = max_x                                      #number of grid cells along the first dimension of the landscape
        self.max_y = max_y                                      #number of grid cells along the second dimension of the landscape
		self.connections = np.array([[3]+[5 for _ in range(max_x-2)] + [3]]+[[5]+[8 for _ in range(max_x-2)] + [5] for _ in range(max_y - 2)] + [[3]+[5 for _ in range(max_x-2)] + [3]])
        self.res_R = res_R                                      #Local optimal growth rate of the resources
        self.res_K = res_K                                      #Local Carrying capacity
		self.population = []
		self.localsizes = np.zeros(max_x, max_y)
		self.initialize_pop()
        
    def initialize_pop(self):
        '''Initialize individuals'''
        startpop = 0.1*self.max_x*self.max_y*self.K #initial metapopulation size
        
        for _ in range(startpop):
            x, y = rnd.randint(0,(self.max_x-1)), rnd.randint(0,(self.max_y-1))
			self.localsizes[x, y] += 1
            self.population.append(Individual(x, y)
                                             
    def lifecycle(self):   
        '''all actions during one timestep for the metapopulation'''
        
        #randomize the order in which individuals will perfom their actions
        rnd.shuffle(self.population)
        
        for ind in self.population:
            ind.age += 1
            #move
            if ind.dispprop*self.connections[ind.x, ind.y]/8 > rnd.random():
				ind.move()
                
            #reproduce
			for _ in range(min(rnd.poisson(2), max(0, K-self.localsizes[x, y]))):
				self.population.append(Individual(ind.x,ind.y)
				self.localsizes[x, y] += 1
			#die
			if ind.age*0.05 > rnd.random():
				self.population.remove(ind)
				self.localsizes[x, y] -= 1
				
def run():
    meta = Metapopulation(dim,dim,R,K)    
    #simulate MAXTIME timesteps (print generation time and metapopulation size for quickly checking during runs)
    for timer in range(MAXTIME): 
        meta.lifecycle()
    print('generation ',timer)
    print("popsize: {}\n".format(len(meta.population)))

############
#Parameters#
############
Maxtime = 10000
R, K=1, 50
dim = 3

run()
print(str(time.clock()))