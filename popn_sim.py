import numpy as np
import pandas as pd
from scipy.stats import expon
import math
import random


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


def split_time(events):
    i = 0
    t = 0
    while (t>=events[i][1]) & (i<len(events)-1):
        n = len(events[i][2])
        t = events[i][1]
        wait = expon.rvs(scale = math.comb(n,2), size = 1)[0]
        t += wait
        i += 1
    if i == len(events)-1:
        t = events[-1][1]+ expon.rvs(1, size = 1)[0]

    return t


def mig_add(nodes, events):
    t_1 = split_time(events)
    t_2 = split_time(events)
    t_from = min(t_1,t_2)
    t_to = max(t_1,t_2)
    t_from_index = timeindex(events,t_from)
    t_to_index = timeindex(events,t_to)

    ##t_from_index is the order index in events.
    if t_from_index == len(events):
        p_from = events[-1][0]
    else:
        p_from = random.sample(events[t_from_index-1][2],1)[0]
    if t_to_index == len(events):
        p_to = events[-1][0]
    else:
        p_to = random.sample(events[t_to_index-1][2],1)[0]
    
    f = random.uniform(0,1)
    new_a, new_b, new_c = max(keys_events(events))+1, max(keys_events(events))+2, max(keys_events(events))+3
    
    #if p_to == p_from:
    #    return nodes, events
    ##This is the case that the migration happens before the root of the original tree.
    if t_from_index==len(events):
        nodes[new_a] = {
        'parents':[new_c],
        'children':[p_from],
        'time':t_from,
        'frac':f
        }
        nodes[new_b] = {
        'parents':[new_c],
        'children':[p_from],
        'time':t_from,
        'frac':1-f
        }
        nodes[new_c] = {
        'parents':[],
        'children':[new_a,new_b],
        'time':t_to,
        'frac':1
        }
        events.append([new_a,t_from,[new_a,new_b],0])
        events.append([new_b,t_from,[new_a,new_b],0])
        events.append([new_c,t_to,[new_c],1])
        nodes[p_from]['parents']=[new_a,new_b]

    ##This is the case that the end of migration is before the root of the tree.
    else:
        if p_to != p_from:

            c_pa = nodes[p_to]['parents']
            c_chi = [new_a, p_to]
            b_pa = nodes[p_from]['parents']

            ##Update the nodes
            nodes[p_to]['parents'] = [new_c]
            nodes[p_from]['parents'] = [new_a,new_b]
            if len(c_pa)>0:
                for x in c_pa:
                    nodes[x]['children'].remove(p_to)
                    nodes[x]['children'].append(new_c)
            if len(b_pa)>0:
                for x in b_pa:
                    nodes[x]['children'].remove(p_from)
                    nodes[x]['children'].append(new_b)
            nodes[new_a] = {
            'parents':[new_c],
            'children':[p_from],
            'time':t_from,
            'frac':f
            }
            nodes[new_b] = {
            'parents':b_pa,
            'children':[p_from],
            'time':t_from,
            'frac':1-f
            }
            nodes[new_c] = {
            'parents':c_pa,
            'children':c_chi,
            'time':t_to,
            'frac':1
            }
            ##Update the events
            events.insert(t_to_index,[new_c,t_to,[],1])
            events.insert(t_from_index,[new_b,t_from,[],0])
            events.insert(t_from_index,[new_a,t_from,[],0])
            for i in range(t_from_index,len(events)):
                events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                if events[i][1] == events[i-1][1]:
                    events[i-1][2].append(events[i][0])
                events[i][2] = list(set(events[i][2]))
        else:
            c_pa = nodes[p_to]['parents']
            nodes[p_from]['parents'] = [new_a,new_b]
            for x in c_pa:
                nodes[x]['children'].remove(p_to)
                nodes[x]['children'].append(new_c)

            nodes[new_a] = {
            'parents':[new_c],
            'children':[p_from],
            'time':t_from,
            'frac':f
            }
            nodes[new_b] = {
            'parents':[new_c],
            'children':[p_from],
            'time':t_from,
            'frac':1-f
            }
            nodes[new_c] = {
            'parents':c_pa,
            'children':[new_a,new_b],
            'time':t_to,
            'frac':1
            }
            events.insert(t_to_index,[new_c,t_to,[],1])
            events.insert(t_from_index,[new_b,t_from,[],0])
            events.insert(t_from_index,[new_a,t_from,[],0])
            for i in range(t_from_index,len(events)):
                events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                if events[i][1] == events[i-1][1]:
                    events[i-1][2].append(events[i][0])
                events[i][2] = list(set(events[i][2]))

    return nodes, events