
import numpy as np
import pandas as pd
from scipy.stats import expon
import math
import random

random.seed(123)

def true_prior(n=4, rho=0.2):
    ##Initiation
    nodes = {}
    for i in range(1,n+1):
        nodes[i] = {
            'parents': [],
            'children': [],
            'time': 0,
            'frac':1
        }
    current = [i for i in range(1,n+1)]
    t = 0
    events = [[0, 0,[i for i in range(1,n+1)],1]]
    while len(current)!=1:
        wait = expon.rvs(scale = 1/(math.comb(len(current),2) + rho*len(current)/2), size = 1)[0]
        t += wait
        prob = random.uniform(0,1)
        banch = (len(current)-1)/(len(current)-1+rho)
        if prob < banch:
            m = random.sample(current, 2)
            new = len(nodes)+1
            nodes[new] = {
                'parents': [],
                'children': m,
                'time': t,
                'frac': 1
            }
            for i in m:
                nodes[i]['parents'] = [new]
            current = list(filter(lambda x: x not in m, current))
            current.append(new)
            events.append([new,t,current,1])
        else:
            f = random.uniform(0,1)
            m = random.sample(current,1)[0]
            new_1 = len(nodes)+1
            new_2 = len(nodes)+2
            nodes[new_1] = {
                'parents': [],
                'children': [m],
                'time':t,
                'frac':f
            }
            nodes[new_2] = {
                'parents': [],
                'children': [m],
                'time': t,
                'frac': 1-f
            }
            nodes[m]['parents'] = [new_1, new_2]
            current = [x for x in current if x!=m]
            current += [new_1, new_2]
            events.append([new_1,t,current,0])
            events.append([new_2,t,current,0])

    return nodes, events




