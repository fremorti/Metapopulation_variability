'''
Created on Feb 27, 2018
Author: fremorti
'''

import random as rnd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-talk')
start = time.clock()
default_path = os.getcwd()

class Individual:
    '''Class that regulates individuals and their properties'''
    def __init__(self,
                 x,
                 y):
        #Initialization
        self.x = x                  #location (x, y)
        self.y = y
        
        
    def move(self, connection, max_x, max_y):
        #an individual moves
        i = rnd.choice(*np.where(connection[self.x*max_y+self.y]))
        self.x, self.y = i//max_y, i%max_y




        
   
class Metapopulation:
    '''Contains the whole population, regulates daily affairs'''
    def __init__(self,
                 max_x,
                 max_y,
				 r,
                 K,
                 d,
                 b):
        """Initialization"""
        self.max_x = max_x                                      #number of grid cells along the first dimension of the landscape
        self.max_y = max_y                                      #number of grid cells along the second dimension of the landscape
        self.fec = r                                      #Local optimal growth rate/optimal fecundity
        self.K = K                                      #Local Carrying capacity
        self.d = d
        self.b = b
        self.population = []
        self.localsize = np.zeros((self.max_x, self.max_y))
        self.initialize_pop()
        self.connection_matrix()
        
    def initialize_pop(self):
        '''Initialize individuals'''
        startpop = 0.5*self.max_x*self.max_y*self.K #initial metapopulation size
        
        for _ in range(int(startpop)):
            x, y = rnd.randint(0,(self.max_x-1)), rnd.randint(0,(self.max_y-1))
            self.population.append(Individual(x, y))
            self.localsize[x, y]+=1


    def connection_matrix(self):
        self.connection = []
        for x in range(self.max_x):
            for y in range(self.max_y):
                conn = [self.max_y*g+h for g in (x-1, x, x+1) for h in (y-1, y, y+1) if (g, h) != (x, y) and 0<=g<self.max_x and 0<=h<self.max_y]

                self.connection.append([(i in conn) for i in range(self.max_x*self.max_y)])
        self.connection = np.array(self.connection)


    def lifecycle(self):   
        '''all actions during one timestep for the metapopulation'''

        self.oldpop = self.population[:]
        self.population = []
        self.oldlocal = self.localsize[:]
        self.localsize = np.zeros((self.max_x, self.max_y))
        crowding_susceptibility =((self.fec**(self.b**-1))-1)/self.K

        rnd.shuffle(self.population)
        for ind in self.oldpop:

            #reproduce
            survival = (1+crowding_susceptibility*int(self.oldlocal[ind.x, ind.y]))**(-self.b)
            for _ in range(np.random.poisson(self.fec)):
                if survival>rnd.random():
                    new = Individual(ind.x, ind.y)
                    #move
                    if self.d > rnd.random():
                        new.move(self.connection, self.max_x, self.max_y)
                    self.population.append(new)
                    self.localsize[new.x, new.y] += 1






class Datacollector:
    def __init__(self, maxtime, max_x, max_y):
        self.mt = maxtime
        self.max_x = max_x
        self.max_y = max_y
        self.sizesintime = np.zeros((maxtime, max_x, max_y))

    def collect(self, time, localsize):
        self.sizesintime[time] = localsize

    def plottotal(self):
        fig, ax = plt.subplots()
        ax.plot(np.sum(self.sizesintime, (1, 2)))
        plt.show()

    def plotlocal(self):
        fig, axes = plt.subplots(self.max_x, self.max_y)
        for i, ax in enumerate(axes.flatten()):
            ax.plot(self.sizesintime[:, i//self.max_y, i%self.max_y])
            ax.set_xlabel('time')
            ax.set_title(i)

        fig.suptitle('local population sizes')
        plt.tight_layout(rect = [0, 0, 1, 0.97])
        plt.show()

    def variabilities(self):
        reshaped_sizes = np.reshape(self.sizesintime, (self.mt, self.max_x*self.max_y))[:]
        total = np.sum(reshaped_sizes, axis = 1)
        total_mean = np.mean(total)
        total_var = np.var(total)/total_mean**2
        covars = np.cov(np.transpose(reshaped_sizes))
        alpha=(np.sum(np.diag(covars)**0.5)/total_mean)**2
        gamma= np.sum(covars)/(total_mean**2)
        beta = alpha-gamma
        return alpha, gamma, beta






def run(r, K, d, b):
    meta = Metapopulation(max_x,max_y,r,K, d, b)
    data = Datacollector(maxtime, max_x, max_y)
    #simulate MAXTIME timesteps (print generation time and metapopulation size for quickly checking during runs)
    for timer in range(maxtime):
        meta.lifecycle()
        data.collect(timer, meta.localsize)
        """print(meta.localsize)
        unique = np.unique([ind.y+max_y*ind.x for ind in meta.population], return_counts = 1)
        print(np.reshape(unique[1], (max_x, max_y)))"""
        #print('generation ',timer)
        #print("popsize: {}\n".format(len(meta.population)))
        #print(f'{meta.localsize[0, 0]}, {meta.localsize[0, 1]}')
    #print(str(time.clock()))
    #data.plotlocal()
    #data.plottotal()
    return data.variabilities()

def replicate_runs(n):
    x_s = []
    a_s = []
    g_s = []
    b_s = []
    for d in np.arange(0.05, 0.55, 0.05):
        for i in range(n):
            x_s+=[d]
            alph, gam, bet = run(r, K, d, b)
            a_s+=[alph]
            g_s+=[gam]
            b_s+=[bet]
            print(f'{d}, replicate {i}')
    print(str(time.clock()))
    data = pd.DataFrame({'disp':x_s,
                        'alphavar':a_s,
                        'gammavar':g_s,
                        'betavar':b_s})
    #data.plot(x = 'disp', y = 'betavar', kind = 'scatter')
    data.boxplot(column='betavar', by = 'disp')
    plt.show()
    """metrics = {'alpha':a_s, 'gamma':g_s, 'beta':b_s}
    for key in metrics:
        fig, ax = plt.subplots()
        ax.plot(x_s, metrics[key], '.')
        ax.set_title = key
        plt.show()"""





############
#Parameters#
############
maxtime = 1000
r, K= 2, 50
b = 1
max_x, max_y = 1, 2
dispprop = 0.25

replicate_runs(20)
#print(run(r, K, dispprop, b))
'''for dispprop in (0.05, 0.2, 0.5):
    for n in range(50):
        a, g, b = run()
        vardic['a'+str(dispprop)] = vardic.get('a'+str(dispprop), [])+[a]
        vardic['g'+str(dispprop)] = vardic.get('g'+str(dispprop), [])+[g]
        vardic['b'+str(dispprop)] = vardic.get('b'+str(dispprop), [])+[b]
        print(f'{dispprop}: rep{n}')

for metric in ('a', 'g', 'b'):
    fig, ax = plt.subplots()
    ax.boxplot([vardic[metric+str(prop)][:10] for prop in (0.05, 0.2, 0.5)], labels = ('0.05', '0.2', '0.5'))
    #ax.set_xlabel = 'dispersal propensity'
    #ax.title = metric
    plt.show()'''

