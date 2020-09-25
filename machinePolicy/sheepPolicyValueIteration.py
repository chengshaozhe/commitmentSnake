import numpy as np
import collections as co
import functools as ft
import itertools as it
import operator as op
import matplotlib.pyplot as plt
import os
import pickle
import sys
sys.setrecursionlimit(2**30)
import pandas as pd
import random
from viz import *
from reward import *


def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(np.array(vector1), np.array(vector2).T)
    innerProduct = vectoriseInnerProduct
    norm1 = computeVectorNorm(vector1)
    norm2 = computeVectorNorm(vector2)
    if norm1 > 0 and norm2 > 0:
        unclipRatio = innerProduct / (norm1 * norm2)
        ratio = np.clip(unclipRatio, -1.0, 1.0)  # float precision probblem as enmin report
        angle = np.arccos(ratio)
    else:
        angle = np.nan
    return angle


def computeVectorNorm(vector):
    L2Norm = np.linalg.norm(vector, ord=2)
    return L2Norm


class HeatSeekingDiscreteDeterministicPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, preyPosition, predatorPosition):
        heatSeekingVector = np.array(preyPosition) - np.array(predatorPosition)
        angleBetweenVectors = {action: computeAngleBetweenVectors(heatSeekingVector, np.array(action)) for action in self.actionSpace}
        optimalActionList = [action for action in angleBetweenVectors.keys() if angleBetweenVectors[action] == min(angleBetweenVectors.values())]
        action = random.choice(optimalActionList)
        return action


class StayWithinBoundary:
    def __init__(self, gridSize, lowerBoundary):
        self.gridX, self.gridY = gridSize
        self.lowerBoundary = lowerBoundary

    def __call__(self, nextIntendedState):
        nextX, nextY = nextIntendedState
        if nextX < self.lowerBoundary:
            nextX = self.lowerBoundary
        if nextX > self.gridX:
            nextX = self.gridX
        if nextY < self.lowerBoundary:
            nextY = self.lowerBoundary
        if nextY > self.gridY:
            nextY = self.gridY
        return nextX, nextY


class Transition:
    def __init__(self, stayWithinBoundary):
        self.stayWithinBoundary = stayWithinBoundary
        self.getAgentsForce = getAgentsForce

    def __call__(self, state, action):
        agentsIntendedState = np.array(state) + np.array(action)
        agentsNextState = [self.stayWithinBoundary(intendedState) for intendedState in agentsIntendedState]
        return agentsNextState


class ValueIteration():
    def __init__(self, gamma, epsilon=0.001, max_iter=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter

    def __call__(self, S, A, T, R):
        gamma, epsilon, max_iter = self.gamma, self.epsilon, self.max_iter
        V_init = {s: 0 for s in S}
        delta = 0
        for i in range(max_iter):
            print(i)
            V = V_init.copy()
            for s in S:
                V_init[s] = max([sum([p * (R[s_n][a] + gamma * V[s_n])
                                      for (s_n, p) in T[s][a].items()]) for a in A])

            delta = np.array([V[s] - V_init[s] for s in S])
            if np.all(delta) < epsilon * (1 - gamma) / gamma:
                break
        return V


class GridWorld():
    def __init__(self, name='', nx=None, ny=None):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.coordinates = tuple(it.product(range(self.nx), range(self.ny)))
        self.terminals = []
        self.obstacles = []
        self.features = co.OrderedDict()

    def add_terminals(self, terminals=[]):
        for t in terminals:
            self.terminals.append(t)

    def add_obstacles(self, obstacles=[]):
        for o in obstacles:
            self.obstacles.append(o)

    def add_feature_map(self, name, state_values, default=0):
        self.features[name] = {s: default for s in self.coordinates}
        self.features[name].update(state_values)

    def is_state_valid(self, state):
        if state[0] not in range(self.nx):
            return False
        if state[1] not in range(self.ny):
            return False
        if state in self.obstacles:
            return False
        return True


def grid_transition(s, a, wolfPolicy=None, is_valid=None):
    sheepState = s[0]
    wolfState = s[1]

    if sheepState == wolfState:
        return {s: 1}
    sheepStateNext = tuple(map(sum, zip(sheepState, a)))
    wolfStateNext = tuple(map(sum, zip(wolfState, wolfPolicy(sheepState, wolfState))))

    if not is_valid(sheepStateNext):
        sheepStateNext = sheepState
    if not is_valid(sheepStateNext):
        wolfStateNext = wolfState

    return {(sheepStateNext, wolfStateNext): 1}


# def grid_reward(sn, a, env=None, const=-1, is_terminal=None):
#     return const + sum(map(lambda f: env.features[f][sn], env.features))

def grid_reward(sn, a, env=None, const=0, is_terminal=None):
    sheepState = sn[0]
    wolfState = sn[1]

    if sheepState == wolfState:
        reward = -100
    else:
        reward = const
    return const + reward


def V_dict_to_array(V):
    V_lst = [V.get(s) for s in S]
    V_arr = np.asarray(V_lst)
    return V_arr


def V_to_Q(V, T=None, R=None, gamma=None):
    V_aug = V[np.newaxis, np.newaxis, :]
    return np.sum(T * (R + gamma * V_aug), axis=2)


if __name__ == '__main__':
    env = GridWorld("test", nx=5, ny=5)
    Q_merge = {}

    numWolf = 2
    wolfState = tuple(it.product(range(env.nx), range(env.ny)))
    wolfStatesAll = list(it.combinations(wolfState, numWolf))

    # wolfStatesAll = tuple(it.product(range(15), repeat=2))

    # sheepValue = {s: 100 for s in wolfStates}
    # env.add_feature_map("goal", sheepValue, default=0)
    # env.add_terminals(list(wolfStates))

    S = tuple(it.product(wolfState, repeat=2))
    A = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))

    wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(A)
    transition_function = ft.partial(
        grid_transition, wolfPolicy=wolfPolicy, is_valid=env.is_state_valid)

    T = {s: {a: transition_function(s, a) for a in A} for s in S}
    T_arr = np.asarray([[[T[s][a].get(s_n, 0) for s_n in S]
                         for a in A] for s in S])

    """set the reward func"""

    # to_sheep_reward = ft.partial(distance_mean_reward, goal=wolfStates, unit=1)
    grid_reward = ft.partial(grid_reward, env=env, const=1)

    func_lst = [grid_reward]

    reward_func = ft.partial(sum_rewards, func_lst=func_lst)

    R = {s: {a: reward_func(s, a) for a in A} for s in S}
    R_arr = np.asarray([[[R[s][a] for s_n in S] for a in A]
                        for s in S], dtype=float)
    # print(R)
    gamma = 0.9

    value_iteration = ValueIteration(gamma, epsilon=0.001, max_iter=100)
    V = value_iteration(S, A, T, R)
    # print(V)

    V_arr = V_dict_to_array(V)
    Q = V_to_Q(V=V_arr, T=T_arr, R=R_arr, gamma=gamma)

    Q_dict = {s: {a: Q[si, ai] for (ai, a) in enumerate(A)} for (si, s) in enumerate(S)}

    for s in S:
        Q_dict[s] = {action: np.divide(Q_dict[s][action], np.sum(list(Q_dict[s].values()))) for action in A}

    print(Q_dict[(1, 1), (3, 3)])
