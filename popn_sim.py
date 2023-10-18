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
        wait = expon.rvs(scale = 1/(math.comb(len(current),2)), size = 1)[0]
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


def split_time(events, t_start=0):
    start = timeindex(events, t_start)
    if start > 0:
        t = start + expon.rvs(scale = 1/(math.comb(len(events[start-1][2]), 2)), size = 1)[0]
    else:
        t = start + expon.rvs(scale = 1/(math.comb(len(events[start][2]), 2)), size = 1)[0]

    while (t>=events[start][1]) & (start<len(events)-1):
        if events[start][1] == events[start+1][1]:
            start += 1
        else:
            n = len(events[start][2])
            t = events[start][1]
            wait = expon.rvs(scale = 1/(math.comb(n,2)), size = 1)[0]
            t += wait
            start += 1
    if (start == len(events)-1) & (start>events[-1][1]):
        t = events[-1][1]+ expon.rvs(1, size = 1)[0]

    return t


def total_length(events):
    total = [0]
    for i in range(len(events)-1):
        if events[i][1] != events[i+1][1]:
            total.append(total[-1] + len(events[i][2])*(events[i+1][1]-events[i][1]))
        else:
            total.append(total[-1])

    return total[-1]


def choose_uniform(events):
    total = [0]
    for i in range(len(events)-1):
        if events[i][1] != events[i+1][1]:
            total.append(total[-1] + len(events[i][2])*(events[i+1][1]-events[i][1]))
        else:
            total.append(total[-1])
    u = random.uniform(0, total[-1])
    i = 0
    while u > total[i]:
        i += 1
    out = random.sample(events[i-1][2],1)[0]
    t = (u - total[i-1])/(len(events[i-1][2])) + events[i-1][1]
    return t, out
    



def mig_add(nodes, events):
    t_from, p_from = choose_uniform(events)
    t_to = split_time(events, t_from)
    t_from_index = timeindex(events,t_from)
    t_to_index = timeindex(events,t_to)
    rjmcmc_ratio = 1/total_length(events) 

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

    return nodes, events, rjmcmc_ratio


def mig_remove(nodes, events):
    ##select all the migration branches
    l = [x for x in events if x[-1]==0]
    p_remove = random.sample(l,1)[0][0]
    #select all the migration branches that removed
    mig_remove = []
    pend = [p_remove]
    margin = nodes[nodes[p_remove]['children'][0]]['parents']
    margin.remove(p_remove)
    mig_count = 1
    while  len(pend)!= 0:
        a = pend.pop()
        mig_remove.append(a)
        ll = nodes[a]['parents']
        if len(ll) == 2:
            pend = pend + ll
            mig_count += 1
        if len(ll) == 1:
            bran = ll[0]
            if bran in margin:
                pend.append(bran)
                margin.remove(bran)
            else:
                margin.append(bran)
    for i in list(nodes.keys()):
        if i in mig_remove:
            del nodes[i]
        else:
            nodes[i]['parents'] = [x for x in nodes[i]['parents'] if x not in mig_remove]
            nodes[i]['children'] = [x for x in nodes[i]['children'] if x not in mig_remove]
    total_remove = mig_remove.copy() + margin.copy()
    lmargin = margin.copy()
    while len(margin)!= 0:
        ##Note that margin can only have one children
        m = margin.pop()
        pa = nodes[m]['parents']
        while nodes[m]['children'][0] in margin:
            m = nodes[m]['children'][0]
            margin.remove(m)
        for x in pa:
            nodes[x]['children'] = nodes[x]['children'] + nodes[m]['children']
        nodes[nodes[m]['children'][0]]['parents'] = pa
    for i in list(nodes.keys()):
        if i in lmargin:
            del nodes[i]
        else:
            nodes[i]['parents'] = [x for x in nodes[i]['parents'] if x not in lmargin]
            nodes[i]['children'] = [x for x in nodes[i]['children'] if x not in lmargin]
    events = [x for x in events if x[0] not in total_remove]
    for i in range(1,len(events)):
            events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
            if events[i][1] == events[i-1][1]:
                events[i-1][2].append(events[i][0])
            events[i][2] = list(set(events[i][2]))
    for i in range(mig_count-1):
         nodes, events, r = mig_add(nodes, events)
         
    
    return nodes, events


def find_current(events, t):
    i = 0
    while (t > events[i][1]) & (i < len(events)):
        i += 1
    if i == len(events):
        return events[-1][2]
    else:
        return events[i-1][2]


def mig_resample(nodes, events):
    l = [x for x in range(1,len(events)-1)]
    r = random.sample(l,1)[0]
    rewire = events[r][0]

    forward_propose = 1
    backward_propose = 1
    
    if len(nodes[rewire]['parents']) == len(nodes[rewire]['parents']):
        ##In this case the rewire has two parents and two children so we do nothing

        forward_propose = 1
        backward_propose = 1
        return nodes, events, forward_propose, backward_propose
    
    elif len(nodes[rewire]['parents']) == 1:
        ##In this case the rewire has one parent then two children, so we reiwre upwards from the birth of rewire

        t_start = nodes[rewire]['time']
        t_end = random.uniform(t_start, events[-1][1])
        
        if t_end < events[-1][1]:
            ##In this case the root does not change

            end_index = timeindex(events, t_end)
            pa = nodes[rewire]['parents'][0]
            original_t = nodes[pa]['time']
            if t_end < original_t:
                p_end = random.sample([x for x in find_current(events,t_end) if x != rewire],1)[0]
                forward_propose *= 1/(len(find_current(events, t_end))-1)
            else:
                p_end = random.sample(find_current(events, t_end),1)[0]
                forward_propose *= 1/len(find_current(events, t_end))

            if p_end != pa:
                ##In this case the topology changed

                if pa == events[-1][0]:
                    ##In this case the parents of rewire is not the root

                    grandpa = nodes[pa]['parents']
                    otherchi = [x for x in nodes[pa]['children'] if x != rewire][0]
                    
                    nodes[otherchi]['parents'] = grandpa
                    for x in grandpa:
                        nodes[x]['children'] = [x for x in nodes[x]['children'] if x != pa] + [otherchi]
                    nodes[pa]['children'] = [rewire, p_end]
                    nodes[pa]['parents'] = nodes[p_end]['parents']
                    for x in nodes[p_end]['parents']:
                        nodes[x]['children'] = [x for x in nodes[x]['children'] if x != p_end] + [pa]
                    nodes[pa]['time'] = t_end
                    nodes[p_end]['parents'] = [pa]

                    del events[timeindex(events,original_t)]
                    if t_end < original_t:
                        events.insert(end_index,[pa,t_end,[],1])
                    else:
                        events.insert(end_index-1,[pa,t_end,[],1])
                    for i in range(1,len(events)):
                        events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                        if events[i][1] == events[i-1][1]:
                            events[i-1][2].append(events[i][0])
                        events[i][2] = list(set(events[i][2]))

                else:
                    ##In this case the parent of rewire is the root, so there's no grandpa

                    otherchi = [x for x in nodes[pa]['children'] if x != rewire][0]
                    if p_end == otherchi:
                        nodes[pa]['time'] = t_end
                        events[-1][1] = t_end
                    else:
                        nodes[otherchi]['parents'] = []
                        nodes[pa]['parents'] = nodes[p_end]['parents']
                        for x in nodes[p_end]['parents']:
                            nodes[x]['children'] = [x for x in nodes[x]['children'] if x != p_end] + [pa]
                        nodes[p_end]['parents'] = [pa]
                        nodes[pa]['time'] = t_end
                        nodes[pa]['children'] = [rewire,p_end]

                        del events[timeindex(events,original_t)]
                        if t_end < original_t:
                            events.insert(end_index,[pa,t_end,[],1])
                        else:
                            events.insert(end_index-1,[pa,t_end,[],1])
                        for i in range(1,len(events)):
                            events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                            if events[i][1] == events[i-1][1]:
                                events[i-1][2].append(events[i][0])
                            events[i][2] = list(set(events[i][2]))
                        
            else:
                ##In this case nodes strucute does not change, only the time order of events changes.

                original_t = nodes[pa]['time']
                nodes[pa]['time'] = t_end
                del events[timeindex(events,original_t)]
                if t_end < original_t:
                    events.insert(end_index,[pa,t_end,[],1])
                else:
                    events.insert(end_index-1,[pa,t_end,[],1])
                startindex = timeindex(events, t_start)
                for i in range(startindex,len(events)):
                        events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                        if events[i][1] == events[i-1][1]:
                            events[i-1][2].append(events[i][0])
                        events[i][2] = list(set(events[i][2]))

        else:
            ##In this case the root is changed to pa

            p_end = events[-1][0]
            pa = nodes[rewire]['parents']
            otherchi = [x for x in nodes[pa]['children'] if x != rewire][0]

            if p_end != pa:
                ## that means pa must has one more parent

                grandpa = nodes[pa]['parents']
                original_t = nodes[pa]['time']
                nodes[p_end]['parents'] = pa
                nodes[pa]['time'] = t_end
                nodes[pa]['parents'] = []
                nodes[pa]['children'] = [p_end, rewire]
                nodes[otherchi]['parents'] = grandpa
                for x in grandpa:
                    nodes[x]['children'] = [x for x in nodes[x]['children'] if x != pa] + [otherchi]
                nodes[rewire]['parents'] = pa

                startindex = timeindex(events, t_start)
                events.insert(len(events),[pa,t_end,[pa],1])
                del events[timeindex(events, original_t)]

                for i in range(startindex,len(events)):
                    events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                    if events[i][1] == events[i-1][1]:
                        events[i-1][2].append(events[i][0])
                    events[i][2] = list(set(events[i][2]))
                
                return nodes, events, forward_propose, backward_propose

            else:
                ## in this case we just update the root's time

                events[-1][1] = t_end
                nodes[p_end]['time'] = t_end
                
                return nodes, events, forward_propose, backward_propose

    else:
        ##in this case, the branch is rewired downwards 
        original_t = nodes[rewire]['time']
        t_start = nodes[nodes[rewire]['parents'][0]]['time']
        t_end = random.uniform(0, t_start)
        end_index = timeindex(events, t_end)
        if len(nodes[rewire]['children']) == 0:
            ##in this case rewire has no children and we do nothing
            forward_propose = 1
            backward_propose = 1
        else:
            ##in this case rewire is a migration branch, we change its children
            chi = nodes[rewire]['children'][0]
            otherpa = [x for x in nodes[chi]['parents'] if x != rewire][0]
            if p_end == otherpa or p_end == chi:
                ##the nodes structure does not change in this case
                nodes[otherpa]['time'] = t_end
                nodes[rewire]['time'] = t_end
                events = [x for x in events if x[0] not in [otherpa, rewire]]
                if t_end < original_t:
                    events.insert(end_index,[rewire,t_end,[],0])
                    events.insert(end_index,[otherpa,t_end,[],0])
                else:
                    events.insert(end_index-2,[rewire,t_end,[],0])
                    events.insert(end_index-2,[otherpa,t_end,[],0])
                for i in range(1,len(events)):
                    events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                    if events[i][1] == events[i-1][1]:
                        events[i-1][2].append(events[i][0])
                    events[i][2] = list(set(events[i][2]))
            else:
                ##rewired to some other branch downwards
                dest_pa = nodes[p_end]['parents']

                nodes[otherpa]['time'] = t_end
                nodes[rewire]['time'] = t_end
                grandpa = nodes[otherpa]['parents']
                nodes[chi]['parents'] = grandpa
                for x in grandpa:
                    nodes[x]['children'] = [chi]
                
                nodes[otherpa]['parents'] = dest_pa
                for x in dest_pa:
                    nodes[x]['children'] = [otherpa]
                nodes[rewire]['children'] = [p_end]
                nodes[otherpa]['children'] = [p_end]
                nodes[p_end]['parents'] = [rewire, otherpa]

                events = [x for x in events if x[0] not in [rewire, otherpa]]
                if t_end < original_t:
                    events.insert(end_index,[rewire,t_end,[],0])
                    events.insert(end_index,[otherpa,t_end,[],0])
                else:
                    events.insert(end_index-2,[rewire,t_end,[],0])
                    events.insert(end_index-2,[otherpa,t_end,[],0])
                for i in range(1,len(events)):
                    events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                    if events[i][1] == events[i-1][1]:
                        events[i-1][2].append(events[i][0])
                    events[i][2] = list(set(events[i][2]))

    return nodes, events,forward_propose, backward_propose


def prior_likelihood(events, rho):
    prod = 1
    for i in range(1,len(events)):
        if events[i][1] == events[i-1][1]:
                prod *= 1
        else:
            rate = math.comb(len(events[i][2]),2) + rho * len(events[i][2])/2
            if events[i][-1] == 1:
                prod *= math.exp(rate)
            else :
                prod *= math.exp(rate)
                prod *= rho/2
    return prod


def mig_rjmcmc(nodes, events):
    old_nodes = nodes.copy()
    old_events = events.copy()

    l = [x for x in events if x[-1]==0]
    ## n is the number of migration events
    n = l/2

    ## with probability 1/4 we add an events, 1/4 we remove an events, and 1/2 resample the coalescent
    ## when there's no events, add events with 1/2 and resample with 1/2
    u = random.uniform(0,1)
    if n == 0:
        if u<1/2:
            nodes, events, propose = mig_resample(nodes, events)
        else:
            nodes, events, r = mig_add(nodes, events)
    

