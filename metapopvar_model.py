'''
Created on Feb 27, 2018
Author: fremorti
'''

import random as rnd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
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
        x_, y_ = self.x, self.y
        while (x_ == self.x and y_ == self.y) or not(0<=self.x<max_x) or not(0<=self.y<max_y):
            self.x = x_ + rnd.choice((-1, 0, 1))
            self.y = y_ + rnd.choice((-1, 0, 1))

        
   
class Metapopulation:
    '''Contains the whole population, regulates daily affairs'''
    def __init__(self,
                 max_x,
                 max_y,
                 r,
				 K,):
        """Initialization"""
        self.max_x = max_x                                      #number of grid cells along the first dimension of the landscape
        self.max_y = max_y                                      #number of grid cells along the second dimension of the landscape
        self.connections = np.array([[3]+[5 for _ in range(max_x-2)] + [3]]+[[5]+[8 for _ in range(max_x-2)] + [5] for _ in range(max_y - 2)] + [[3]+[5 for _ in range(max_x-2)] + [3]])
        self.r = r                                      #Local optimal growth rate of the resources
        self.K = K                                      #Local Carrying capacity
        self.mort = 0.1
        self.population = []
        self.localsize = np.zeros((max_x, max_y))
        self.initialize_pop()
        
    def initialize_pop(self):
        '''Initialize individuals'''
        startpop = self.max_x*self.max_y*self.K #initial metapopulation size
        
        for _ in range(int(startpop)):
            x, y = rnd.randint(0,(self.max_x-1)), rnd.randint(0,(self.max_y-1))
            self.localsize[x, y] += 1
            self.population.append(Individual(x, y))
                                             
    def lifecycle(self):   
        '''all actions during one timestep for the metapopulation'''
        
        #randomize the order in which individuals will perfom their actions
        rnd.shuffle(self.population)
        
        for ind in self.population:
            ind.age += 1
            #move
            if ind.dispprop*self.connections[ind.x, ind.y]/8 > rnd.random():
                self.localsize[ind.x, ind.y]-=1
                ind.move(self.max_x, self.max_y)
                self.localsize[ind.x, ind.y]+=1
            """
            #reproduce
            for _ in range(np.random.poisson(max(0, self.mort + self.r*(self.K/self.localsize[ind.x, ind.y]-1)))):
                self.population.append(Individual(ind.x,ind.y))
                self.localsize[ind.x, ind.y] += 1
            #die
            if self.mort > rnd.random():
                self.population.remove(ind)
                self.localsize[ind.x, ind.y] -= 1"""


class Datacollector:
    def __init__(self, maxtime, xmax, ymax):
        self.mt = maxtime
        self.xmax = xmax
        self.ymax = ymax
        self.sizesintime = np.zeros((maxtime, xmax, ymax))

    def collect(self, time, localsize):
        self.sizesintime[time] = localsize

    def plottotal(self):
        fig, ax = plt.subplots()
        ax.plot(np.sum(self.sizesintime, (1, 2)))
        plt.show()

    def plotlocal(self):
        fig, ax = plt.subplots(self.xmax, self.ymax)
        for i in range(self.ymax):
            for j in range(self.xmax):
                ax[i, j].plot(self.sizesintime[:, i, j])
                ax[i, j].set_xlabel('time')

        fig.suptitle('local population sizes')
        plt.tight_layout(rect = [0, 0, 1, 0.97])
        plt.show()

    def variabilities(self):
        reshaped_sizes = np.reshape(self.sizesintime, (self.mt, self.xmax*self.ymax))[-20:]
        total = np.sum(reshaped_sizes, axis = 1)
        total_mean = np.mean(total)
        total_var = np.var(total)/total_mean**2
        covars = np.cov(np.transpose(reshaped_sizes))
        alpha=(np.sum(np.diag(covars)**0.5)/total_mean)**2
        gamma= np.sum(covars)/total_mean**2
        beta = alpha/gamma
        print(f'shaopeng gamma: {gamma}\nvariance of metapopulationsize: {total_var}\nmean: {total_mean}')






def run():
    meta = Metapopulation(dim,dim,r,K)
    data = Datacollector(maxtime, dim, dim)
    #simulate MAXTIME timesteps (print generation time and metapopulation size for quickly checking during runs)
    for timer in range(maxtime):
        meta.lifecycle()
        data.collect(timer, meta.localsize)
        print('generation ',timer)
        print("popsize: {}\n".format(len(meta.population)))

    #data.plottotal()
    data.plotlocal()
    data.variabilities()

############
#Parameters#
############
maxtime = 100
r, K= 1, 200
dim = 3

run()
print(str(time.clock()))