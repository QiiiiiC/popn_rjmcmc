
import numpy as np
import pandas as pd
from scipy.stats import expon
import math
import random

random.seed(123)
def popn_create(n=4):
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
        wait = expon.rvs(scale = math.comb(len(current),2), size = 1)[0]
        t += wait
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
        

    return nodes,events

def read_events(events):
    out = {}
    for i in events:
        out[i[0]] = {'time': i[1],
                     'current': i[2],
                     'merge':i[3],
                     }
    return out

def keys_events(events):
    out = []
    for i in events:
        out.append(i[0])
    return out

def timeindex(events,t):
    i = 0
    if t>events[-1][1]:
        return len(events)
    else:
        while t > events[i][1]:
            i = i+1
    return i

n,e = popn_create()
    
print(e)
print(n)

t_from = 1
t_to = 4.5
t_from_index = timeindex(e,t_from)
t_to_index = timeindex(e,t_to)
if t_from_index == len(e):
        p_from = e[-1][0]
else:
        p_from = random.sample(e[t_from_index-1][2],1)[0]
if t_to_index == len(e):
        p_to = e[-1][0]
else:
        p_to = random.sample(e[t_to_index-1][2],1)[0]
    
f = random.uniform(0,1)
new_a, new_b, new_c = max(keys_events(e))+1, max(keys_events(e))+2, max(keys_events(e))+3

c_pa = n[p_to]['parents']
c_chi = [new_a,p_to]
b_pa = n[p_from]['parents']
n[p_to]['parents'] = [new_c]
n[p_from]['parents'] = [new_a,new_b]
for x in c_pa:
    n[x]['children'].remove(p_to)
    n[x]['children'].append(new_c)
for x in b_pa:
    n[x]['children'].remove(p_from)
    n[x]['children'].append(new_b)
n[new_a] = {
        'parents':[new_c],
        'children':[p_from],
        'time':t_from,
        'frac':f
        }
n[new_b] = {
        'parents':b_pa,
        'children':[p_from],
        'time':t_from,
        'frac':1-f
        }
n[new_c] = {
        'parents':c_pa,
        'children':c_chi,
        'time':t_to,
        'frac':1
        }
e.insert(t_to_index,[new_c,t_to,[],1])
e.insert(t_from_index,[new_b,t_from,[],0])
e.insert(t_from_index,[new_a,t_from,[],0])
for i in range(t_from_index,len(e)):
    e[i][2] = [x for x in e[i-1][2] if x not in n[e[i][0]]['children']]+[e[i][0]]
    if e[i][1] == e[i-1][1]:
        e[i-1][2].append(e[i][0])
e