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
    if (start == len(events)-1) & (t>events[-1][1]):
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
    

def up_mig_id(nodes, events):
    l = []
    for i in range(len(events)):
        if (events[i][-1] == 0) and (len(nodes[events[i][0]]['parents']) == 1):
            l.append(events[i][0])
    return l


def mig_add(nodes, events, rho):
    t_from, p_from = choose_uniform(events)
    t_to = split_time(events, t_from)
    t_from_index = timeindex(events,t_from)
    t_to_index = timeindex(events,t_to)
    reverse = len(up_mig_id(nodes, events))
    rjmcmc_ratio = rho/2 * total_length(events) * math.exp(rho/2*(t_to - t_from))/(reverse + 2)

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





def mig_remove(nodes, events, rho):
    l = up_mig_id(nodes, events)
    p_remove = random.sample(l,1)[0]
    t_1 = nodes[p_remove]['time']
    T = total_length(events)
    if nodes[p_remove]['time'] == nodes[[x for x in nodes[nodes[p_remove]['parents'][0]]['children'] if x != p_remove][0]]['time']:
        pa = nodes[p_remove]['parents'][0]
        t_2 = nodes[pa]['time']
        grandpa = nodes[pa]['parents']
        chi = nodes[p_remove]['children'][0]
        otherpa = nodes[chi]['parents']
        nodes[chi]['parents'] = grandpa
        for x in grandpa:
            nodes[x]['children'] = [x for x in nodes[x]['children'] if x != pa] + [chi]
        
        b = []
        total = [pa] + otherpa
        nodes = dict((k, v) for k,v in nodes.items() if k not in total)
        events = [x for x in events if x[0] not in total]
    else:
        pa = nodes[p_remove]['parents'][0]
        t_2 = nodes[pa]['time']
        pachi = [x for x in nodes[pa]['children'] if x != p_remove][0]
        grandpa = nodes[pa]['parents']
        nodes[pachi]['parents'] = grandpa
        for x in grandpa:
            nodes[x]['children'] = [x for x in nodes[x]['children'] if x != pa] + [pachi]

        ppachi = nodes[p_remove]['children'][0]
        ppa = [x for x in nodes[ppachi]['parents'] if x != p_remove][0]
        grandppa = nodes[ppa]['parents']
        nodes[ppachi]['parents'] = grandppa
        for x in grandppa:
            nodes[x]['children'] = [x for x in nodes[x]['children'] if x != ppa] + [ppachi]
        nodes = dict((k, v) for k,v in nodes.items() if k not in [pa, p_remove, ppa])

        events = [x for x in events if x[0] not in [pa, p_remove, ppa]]

    for i in range(1, len(events)):
        events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
        if events[i][1] == events[i-1][1]:
            events[i-1][2].append(events[i][0])
        events[i][2] = list(set(events[i][2]))

    rjmcmc_ratio = 2 * len(l)/(rho * T) / math.exp(rho/2*(t_2-t_1))

    return nodes, events, rjmcmc_ratio


def find_current(events, t):
    i = 0
    while (t > events[i][1]) & (i < len(events)):
        i += 1
    if i == len(events):
        return events[-1][2]
    else:
        return events[i-1][2]

def length_between(events,t_1, t_2):
    l = 0
    if t_2 < events[-1][1]:
        start_index = timeindex(events, t_1)
        end_index = timeindex(events,t_2)
        if start_index == end_index:
            l += (t_2-t_1) * len(events[start_index-1][2])
        else:
            l += (events[start_index][1]-t_1) * len(events[start_index-1][2])
            for i in range(start_index,end_index):
                l += (events[i+1][1]-events[i][1]) * len(events[i][2])
            l += (t_2 - events[end_index][1]) * len(events[end_index][2])
    elif t_1 < events[-1][1]:
        start_index = timeindex(events, t_1)
        l += (events[start_index][1]-t_1) * len(events[start_index-1][2])
        for i in range(start_index+1,len(events)):
            l += (events[i][1]-events[i][1]) * len(events[i-1][2])
        l += (t_2 - events[-1][1])
    else:
        l += t_2 - t_1

    return l



def resample_up_time(nodes, events, rewire, starttime):
    start = timeindex(events, starttime)
    a = nodes[rewire]['parents']
    if len(a) == 0:
        l = len(events)-1
    else:
        l = timeindex(events,nodes[a[0]]['time'])

    k = len(events[start][2])
    if k == 2:
        t = start + expon.rvs(scale = 1, size = 1)[0]
    else:
        t = start + expon.rvs(scale = 1/math.comb(k-1,2), size = 1)[0]

    while (t>=events[start][1]) & (start<len(events)-1):
        if events[start][1] == events[start+1][1]:
            start += 1
        else:
            if start < l:
                n = len(events[start][2])-1
            else:
                n = len(events[start][2])
            if n == 1:
                n = 2
            t = events[start][1]
            wait = expon.rvs(scale = 1/(math.comb(n,2)), size = 1)[0]
            t += wait
            start += 1
    if (start == len(events)-1) & (t>events[-1][1]):
        t = events[-1][1]+ expon.rvs(1, size = 1)[0]

    return t

def mig_resample(nodes, events):
    # oldnodes = nodes.copy()
    # oldevents = events.copy()
    l = [x for x in nodes.keys() if x != events[-1][0]]
    rewire = random.sample(l,1)[0]
    rjmcmc_ratio = 1
    
    if len(nodes[rewire]['parents']) == len(nodes[rewire]['parents'])==2:
        ##In this case the rewire has two parents and two children

        return nodes, events, 1
    
    elif len(nodes[rewire]['parents']) == 1:
        ##In this case the rewire has one parent then two children, so we reiwre upwards from the birth of rewire

        t_start = nodes[rewire]['time']
        t_end = resample_up_time(nodes, events,rewire, t_start)
        
        if t_end < events[-1][1]:
            ##In this case the root does not change

            pa = nodes[rewire]['parents'][0]
            original_t = nodes[pa]['time']
            if t_end < original_t:
                p_end = random.sample([x for x in find_current(events,t_end) if x != rewire],1)[0]
                rjmcmc_ratio = math.exp(-length_between(events,t_end, original_t) + t_end - t_start)
            else:
                p_end = random.sample(find_current(events, t_end),1)[0]
                rjmcmc_ratio = math.exp(-length_between(events,original_t, t_end) + original_t - t_start)
            if p_end != pa:
                ##In this case the topology changed

                if pa != events[-1][0]:
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

                    events = [x for x in events if x[0] != pa]
                    events.insert(timeindex(events,t_end),[pa,t_end,[],1])
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

                        events = [x for x in events if x[0] != pa]
                        events.insert(timeindex(events,t_end),[pa,t_end,[],1])
                        for i in range(1,len(events)):
                            events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                            if events[i][1] == events[i-1][1]:
                                events[i-1][2].append(events[i][0])
                            events[i][2] = list(set(events[i][2]))
                        
            else:
                ##In this case nodes strucute does not change, only the time order of events changes.
                nodes[pa]['time'] = t_end
                events = [x for x in events if x[0] != pa]
                events.insert(timeindex(events,t_end),[pa,t_end,[],1])
                for i in range(1,len(events)):
                        events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                        if events[i][1] == events[i-1][1]:
                            events[i-1][2].append(events[i][0])
                        events[i][2] = list(set(events[i][2]))

        else:
            ##In this case the root is changed to pa

            p_end = events[-1][0]
            pa = nodes[rewire]['parents'][0]
            otherchi = [x for x in nodes[pa]['children'] if x != rewire][0]

            if p_end != pa:
                ## that means pa must has one layer of parent

                grandpa = nodes[pa]['parents']
                nodes[p_end]['parents'] = [pa]
                nodes[pa]['time'] = t_end
                nodes[pa]['parents'] = []
                nodes[pa]['children'] = [p_end, rewire]
                nodes[otherchi]['parents'] = grandpa
                for x in grandpa:
                    nodes[x]['children'] = [x for x in nodes[x]['children'] if x != pa] + [otherchi]
                nodes[rewire]['parents'] = [pa]

                events = [x for x in events if x[0] != pa]
                events.append([pa,t_end,[],1])

                for i in range(1,len(events)):
                    events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
                    if events[i][1] == events[i-1][1]:
                        events[i-1][2].append(events[i][0])
                    events[i][2] = list(set(events[i][2]))
                
                return nodes, events, rjmcmc_ratio

            # else:
            #     ## in this case we just update the root's time

            #     events[-1][1] = t_end
            #     nodes[p_end]['time'] = t_end
                
            #     return nodes, events, rjmcmc_ratio

    # else:
    #     ##in this case, the branch is rewired downwards 
    #     original_t = nodes[rewire]['time']
    #     t_start = nodes[nodes[rewire]['parents'][0]]['time']
    #     t_end = random.uniform(0, t_start)
    #     end_index = timeindex(events, t_end)
    #     if len(nodes[rewire]['children']) == 0:
    #         ##in this case rewire has no children
    #         pa1 = nodes[rewire]['parents'][0]
    #         pa2 = nodes[rewire]['parents'][1]
    #         t_stop = min(nodes[nodes[pa1]['parents'][0]]['time'], nodes[nodes[pa2]['parents'][0]]['time'])
    #         t_end = random.uniform(0,t_stop)
    #         for x in nodes[rewire]['parents']:
    #             nodes[x]['time'] = t_end
    #         for i in range(1,len(events)):
    #                 events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
    #                 if events[i][1] == events[i-1][1]:
    #                     events[i-1][2].append(events[i][0])
    #                 events[i][2] = list(set(events[i][2]))
    #         rjmcmc_ratio = 1

    #     else:
    #         ##in this case rewire is a migration branch, we change its children
    #         chi = nodes[rewire]['children'][0]
    #         otherpa = [x for x in nodes[chi]['parents'] if x != rewire][0]
    #         if p_end == otherpa or p_end == chi:
    #             ##the nodes structure does not change in this case
    #             nodes[otherpa]['time'] = t_end
    #             nodes[rewire]['time'] = t_end
    #             events = [x for x in events if x[0] not in [otherpa, rewire]]
    #             if t_end < original_t:
    #                 events.insert(end_index,[rewire,t_end,[],0])
    #                 events.insert(end_index,[otherpa,t_end,[],0])
    #             else:
    #                 events.insert(end_index-2,[rewire,t_end,[],0])
    #                 events.insert(end_index-2,[otherpa,t_end,[],0])
    #             for i in range(1,len(events)):
    #                 events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
    #                 if events[i][1] == events[i-1][1]:
    #                     events[i-1][2].append(events[i][0])
    #                 events[i][2] = list(set(events[i][2]))
    #         else:
    #             ##rewired to some other branch downwards
    #             dest_pa = nodes[p_end]['parents']

    #             nodes[otherpa]['time'] = t_end
    #             nodes[rewire]['time'] = t_end
    #             grandpa = nodes[otherpa]['parents']
    #             nodes[chi]['parents'] = grandpa
    #             for x in grandpa:
    #                 nodes[x]['children'] = [chi]
                
    #             nodes[otherpa]['parents'] = dest_pa
    #             for x in dest_pa:
    #                 nodes[x]['children'] = [otherpa]
    #             nodes[rewire]['children'] = [p_end]
    #             nodes[otherpa]['children'] = [p_end]
    #             nodes[p_end]['parents'] = [rewire, otherpa]

    #             events = [x for x in events if x[0] not in [rewire, otherpa]]
    #             if t_end < original_t:
    #                 events.insert(end_index,[rewire,t_end,[],0])
    #                 events.insert(end_index,[otherpa,t_end,[],0])
    #             else:
    #                 events.insert(end_index-2,[rewire,t_end,[],0])
    #                 events.insert(end_index-2,[otherpa,t_end,[],0])
    #             for i in range(1,len(events)):
    #                 events[i][2] = [x for x in events[i-1][2] if x not in nodes[events[i][0]]['children']]+[events[i][0]]
    #                 if events[i][1] == events[i-1][1]:
    #                     events[i-1][2].append(events[i][0])
    #                 events[i][2] = list(set(events[i][2]))
    #             if t_end < original_t:
    #                 rjmcmc_ratio = len(find_current(events,original_t))/len(find_current(events, t_end))
    #             else:
    #                 rjmcmc_ratio = len(find_current(events,original_t))/(len(find_current(events, t_end))-1)
    # r = False
    # for i in range(len(events)):
    #     if i < len(events)-1:
    #         if len(events[i][2]) == 1:
    #             r = True
    # if r:
    #     return oldnodes, oldevents,1

    return nodes, events,rjmcmc_ratio


def prior_likelihood(events, rho):
    prod = 1
    for i in range(1,len(events)):
        if events[i][1] == events[i-1][1]:
                prod *= 1
        else:
            rate = -(math.comb(len(events[i][2]),2) + rho * len(events[i][2])/2)
            if events[i][-1] == 1:
                prod *= math.exp(rate * (events[i][1]- events[i-1][1]))
            else :
                prod *= math.exp(rate * (events[i][1]- events[i-1][1]))
                prod *= rho/2
    return prod


def mig_rjmcmc(nodes, events,rho):
    l = [x for x in events if x[-1]==0]
    ## with probability 1/4 we add an events, 1/4 we remove an events, and 1/2 resample the coalescent
    ## when there's no events, add events with 1/2 and resample with 1/2
    u = random.uniform(0,1)
    if len(l) == 0:
        if u<1/2:
            newnodes, newevents, rjmcmc_ratio = mig_resample(nodes, events)
            u1 = random.uniform(0,1)
            if u1 < min(1, prior_likelihood(newevents, rho)/prior_likelihood(events, rho)*rjmcmc_ratio):
                for i in range(len(events)):
                    if (i < len(events)-1) and (len(events[i][2]) == 1):
                        nodes, events = nodes, events
                    else:
                        r = False
                        for i in range(len(events)):
                            if (i < len(events)-1) and (len(events[i][2]) == 1):
                                r = True
                        if r:
                            nodes, events = nodes, events
                        else:
                            nodes, events = newnodes, newevents
        else:
            newnodes, newevents, rjmcmc_ratio = mig_add(nodes, events,rho)
            u1 = random.uniform(0,1)
            if u1 < min(1,rjmcmc_ratio):
                r = False
                for i in range(len(events)):
                    if (i < len(events)-1) and (len(events[i][2]) == 1):
                        r = True
                if r:
                    nodes, events = nodes, events
                else:
                    nodes, events = newnodes, newevents
    else:
        if u < 1/4:
            newnodes, newevents, rjmcmc_ratio = mig_add(nodes, events, rho)
            u1 = random.uniform(0,1)
            if u1 < min(1,rjmcmc_ratio):
                r = False
                for i in range(len(events)):
                    if (i < len(events)-1) and (len(events[i][2]) == 1):
                        r = True
                if r:
                    nodes, events = nodes, events
                else:
                    nodes, events = newnodes, newevents
        if u > 3/4:
            newnodes, newevents, rjmcmc_ratio = mig_remove(nodes, events, rho)
            u1 = random.uniform(0,1)
            if u1 < min(1, rjmcmc_ratio):
                r = False
                for i in range(len(events)):
                    if (i < len(events)-1) and (len(events[i][2]) == 1):
                        r = True
                if r:
                    nodes, events = nodes, events
                else:
                    nodes, events = newnodes, newevents
                
        else:
            newnodes, newevents, rjmcmc_ratio = mig_resample(nodes, events)
            u1 = random.uniform(0,1)
            if u1 < min(1, prior_likelihood(newevents, rho)/prior_likelihood(events, rho)*rjmcmc_ratio):
                r = False
                for i in range(len(events)):
                    if (i < len(events)-1) and (len(events[i][2]) == 1):
                        r = True
                if r:
                    nodes, events = nodes, events
                else:
                    nodes, events = newnodes, newevents
    return nodes, events
    

nodes, events = popn_create()
k = [events[-1][1]]
for i in range(500):
    nodes, events = mig_rjmcmc(nodes, events,0.2)
    k.append(events[-1][1])
sum(k)/len(k)

def check_single(nodes):
    for i in nodes:
        if len(nodes[i]['parents']) == len(nodes[i]['children'])==1:
            if len(nodes[nodes[i]['parents'][0]]['children']) ==1:
                return False
    return True

def check_time_order(events):
    r = True
    for i in range(1,len(events)):
        if events[i][1] < events[i-1][1]:
            r = False
    return r

nodes, events = popn_create()
nodes, events, i = mig_add(nodes, events)
check_single(nodes)

nodes, events = popn_create()
for i in range(10):
    nodes, events, r = mig_add(nodes, events, 0.2)
for i in range(100):
    nodes, events, r = mig_remove(nodes, events, 0.2)
nodes, events = popn_create()
for i in range(1):
    nodes, events, r = mig_add(nodes, events, 0.2)
for i in range(50000):
    nodes, events, r = mig_resample(nodes, events)
events
