import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import math


class Structure:
    """ Represents a Resident with specific attributes related to social cost and subsidy. """

    def __init__(self, subsidy, EAD, delta, index, h, a, resident_discount_rate):
        self.index = index
        self.h = h
        self.subsidy = subsidy
        self.EAD = EAD
        self.delta = delta
        self.resident_discount_rate = resident_discount_rate
        self.a = 0
        self.prob_accepting = 0.5
        self.strategy = {}

    def update_a(self):
        """ Update acceptance counter. """
        self.a = 1


class Grid:
    """ Represents a Grid with characteristics related to vacancy compounding and resident objectives. """

    def __init__(self, na, nd, v0, r):
        self.na = na
        self.nd = nd
        self.v0 = v0
        self.r = r
        self.N = self.na + self.nd  # Total number of people in the neighborhood.
        self.gamma = 1 / (1 + self.r)  # Resident discount factor.

    def update_B(self):
        """ Update the vacancy factor B. """
        if self.na == self.nd == 0:
            self.B = 0.5
        else:
            self.B = 0.5 + (self.nd / (self.na + self.nd))

    def vacancy_compounding_factor(self, t, v_initial=None):
        """ Calculate the vacancy compounding factor using a sigmoid curve. """
        v_initial = v_initial if v_initial is not None else self.v0
        B = 0.5 if self.na == 0 else 0.5 + (self.nd / (self.na + self.nd))
        v_t = self.na / (1 + v_initial * np.exp(-B * t))**(1/2)
        return v_t


class Government:
    """ Represents the Government that tracks subsidies, relocation, and objective costs. """

    def __init__(self, disMethod, disRate, alpha):
        self.disMethod = disMethod
        self.disRate = disRate
        self.alpha = alpha
        self.pastloss = {}
        self.Subsidyyear = {}

    def discounted_past_loss(self, residents, total_length):
        """ Calculate discounted past loss for each resident. """
        for resident in residents:
            self.pastloss[resident.index] = {}
            for i in range(total_length):
                discounted_loss = 0
                if self.disMethod == "Exponential":
                    for j in range(i):
                        discounted_loss += resident.EAD[j] / (1 + self.disRate) ** j
                elif self.disMethod == 'Hyperbolic':
                    for j in range(i):
                        discounted_loss += resident.EAD[j] / (1 + self.alpha * j)
                self.pastloss[resident.index][i] = discounted_loss


class Oracle:
    """ Oracle to track optimal strategies and utilities. """

    def __init__(self, disc_rate, utility, obj_func):
        self.disc_rate = disc_rate
        self.utility = utility
        self.obj_func = obj_func

    def oracle_utility(self, n_a, n_d, structures, gamma_O, grd):
        """ Calculate Oracle's total utility U_O(π_n). """
        total_utility = 0
        for t in range(20):  # Sum over time t = 1 to 20
            for structure in structures:
                if n_a > 0:
                    total_utility += gamma_O**t * (np.sum(structure.EAD) + grd.vacancy_factor(n_a, n_d) * structure.community_cost)
                else:
                    total_utility += gamma_O**t * np.sum(structure.EAD)
        return total_utility

    def objective_function(self, strategies, structures, gamma_O, grd):
        """ Implements Oracle's objective function. """
        total_objective = 0
        for strategy in strategies.values():
            probability = 1 / len(strategies)
            utility = self.oracle_utility(strategy['n_a'], strategy['n_d'], structures, gamma_O, grd)
            total_objective += probability * utility
        return total_objective


def load_data():
    """ Load data from CSV files. """
    resident_info = pd.read_csv("cost_replacement_relocation.csv")
    ead_info = pd.read_csv('EAD_g500_interpolated.csv')
    merged_df = pd.merge(resident_info, ead_info, on='structure_id', how='left', validate='1:1')
    merged_df.fillna(0, inplace=True)
    resident_info = merged_df[['structure_id', 'replacement_cost', 'relocation_cost', 'mhi_ratio']]
    ead_info_new = merged_df.drop(['replacement_cost', 'relocation_cost', 'mhi_ratio'], axis=1)
    ead_info_new.fillna(0, inplace=True)
    return resident_info, ead_info_new


def main():
    """ Main function to execute the refactored logic. """
    resident_info, ead_info_new = load_data()

    num_structures = 10000
    num_rounds = 20
    num_episodes = 1000

    r = 0.05  # Resident discount rate
    gamma = 1 / (1 + r)
    v0 = 0.1  # Initial vacancy factor

    # Homeowner attributes
    np.random.seed(42)
    zeta = np.random.uniform(50, 200, num_structures)
    S = np.random.uniform(0, 50, num_structures)
    C = np.random.uniform(10, 100, num_structures)
    EAD = np.random.uniform(1000, 5000, num_structures)

    policy_net = PolicyNetwork(4, 4)
    value_net = ValueNetwork(4)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)

    # Example main loop
    for episode in range(num_episodes):
        state = np.random.rand(4)
        action = np.random.choice([0, 1])  # Random action
        reward = np.random.rand()
        next_state = state + 0.1 * (np.random.rand(4) - 0.5)
        advantage = reward + gamma * reward - reward

        policy_net.train()
        policy_optimizer.zero_grad()
        value_net.train()
        value_optimizer.zero_grad()
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Training in progress...")


if __name__ == '__main__':
    main()
