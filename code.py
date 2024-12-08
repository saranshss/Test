import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import math


class Structure:
"""
Represents a Resident with specific attributes related to social cost and subsidy.
"""
def __init__(self, subsidy, EAD, delta, index, h, a, resident_discount_rate):
    """
    Initialize a Resident instance.

    :param zeta: float - The social cost associated with the resident.
    :param subsidy_i: float - The subsidy offered to the resident.
    :delta : privately calculated damage information
    :h : flood depth
    : a: float - acceptance counter
    : prob_accepting : float - probability of accepting
    :resident_discount_rate : float - resident discount rate
    :community_cost : float - community cost
    """
    self.index = index
    self.h = h
    self.zeta = zeta
    self.zeta_o = zeta_o
    self.subsidy = subsidy
    
    self.community_cost = community_cost
    self.a = 0
    self.EAD = EAD
    self.delta = EAD
    self.prob_accepting = 0.5
    self.resident_discount_rate =  resident_discount_rate
    self.utility = utility
    self.obj = obj
    self.strategy = {}
    

def __repr__(self):
    return f"Structure(Social Cost: {self.zeta}, Subsidy Offered: {self.subsidy_i})"

def update_a() :
    a = 1


# Function to calculate disutility for a resident
def utility_structure(i, action, n_a, n_d, grd):
    if action == 1:
        return zeta[i] - S[i]  # Accept action utility
    else:  # 'decline' action
        return delta if ((n_a == 0) and (n_d == 0)) else (delta + grd.vacancy_factor(n_a, n_d) * community_cost)

def u_o() :
  if action == 1:
      return np.random.lognormal(mean=0, sigma=1) - S[i]  # Accept action utility
  else:  # 'decline' action
      return ead if ((n_a == 0) and (n_d == 0)) else (ead + grd.vacancy_factor(n_a, n_d) * community_cost)



class Grid:
    """
    Represents a Grid with characteristics related to vacancy compounding and resident objectives.
    """
    def __init__(self, na, nd, v0, r):
        """
        Initialize a Grid instance.

        :param na: int - Number of active residents.
        :param nd: int - Number of dormant residents.
        :param v0: float - Initial vacancy compounding factor.
        :param r: float - Resident discount rate.
        """
        self.na = na
        self.nd = nd
        self.v0 = v0
        self.r = r
        self.N = self.na + self.nd  # Total number of people in the neighborhood.
        self.gamma = 1 / (1 + self.r)  # Resident discount factor.
        self.policy = policy

    def vacancy_compounding_factor(self, t, v_initial=None):
        """
        Calculates the vacancy compounding factor using a sigmoid curve.

        :param t: float - Time period.
        :param v_initial: float - Initial vacancy factor, if not provided uses self.v0.
        :return: float - Vacancy compounding factor.
        """
        v_initial = v_initial if v_initial is not None else self.v0
        B = 0.5 if self.na == 0 else 0.5 + (self.nd / (self.na + self.nd))
        v_t = self.na / (1 + v_initial * math.exp(-B * t))^(1/nu)
        return v_t

    def resident_objective_function(self, utility, action_costs, state_costs):
        """
        Computes the resident's objective function.

        :param utility: function - Utility function u_g(s_k).
        :param action_costs: list - Action costs a_k for each time step.
        :param state_costs: list - State costs s_k for each time step.
        :return: float - Minimized objective function value.
        """
        if len(action_costs) != len(state_costs):
            raise ValueError("Action costs and state costs must have the same length.")

        total_cost = 0
        for t, (a_k, s_k) in enumerate(zip(action_costs, state_costs)):
            u_sk = utility(s_k)
            if u_sk <= utility(s_k - 1):  # Constraint u_g(s_k) ≤ u_g(s_k-1).
                total_cost += (self.gamma ** t) * utility(a_k)
        return total_cost



class government:
    def __init__(self, disMethod, disRate, alpha):
        self.disMethod = disMethod
        self.disRate = disRate
        self.alpha = alpha

        self.subPercent = 0.5

        # record the past loss for each resident at each year
        self.pastloss = {}
        self.NPVlossSubsidy = {}
        self.Subsidyyear = {}

        # lists to record the number of relocation each year
        self.selfRelocationNum = []
        self.motiRelocationNum = []
        self.optMotiRelocationNum = []

        # lists to record the objective of three different mode
        self.objective_wo_subsidy = 0
        self.objective_fixed_subsidy = 0
        self.obj_fixed_subsidy_replacement = 0

        self.objective_optimize_individually = 0
        self.obj_optimal_subsidy_replacement = 0

    # A function to calculate the discounted
    def discountedPastLoss(self, residents, totalLength):
        for resident in residents:
            self.pastloss[resident.idx] = {}
            for i in range(totalLength):
                discountedLoss = 0
                if self.disMethod == "Exponential":
                    for j in range(0, i):
                        discountedLoss += resident.ead[j] / (1 + self.disRate) ** j
                elif self.disMethod == 'Hyperbolic':
                    for j in range(0, i):
                        discountedLoss += resident.ead[j] / (1 + self.alpha * j)
                else:
                    print("Please enter a valid discounting method!")
                self.pastloss[resident.idx][i] = discountedLoss


class Oracle() :
    def __init__(self, disc_rate, utility, obj_func):
      self.disc_rate = disc_rate
      self.utility = utility
      self.obj_func = obj_func

    def oracle_utility(self, n_a, n_d, structures, gamma_O, grd):
        """
        Calculate Oracle's total utility U_O(π_n).
        
        :param n_a: int - Number of accepted residents.
        :param n_d: int - Number of declined residents.
        :param structures: list - List of Structure objects.
        :param gamma_O: float - Oracle's discount factor.
        :param grd: GRD object - Used to calculate the vacancy factor.
        :return: float - The Oracle's total utility U_O(π_n).
        """
        total_utility = 0
        for t in range(20):  # Sum over time t = 1 to 20
            for structure in structures:
                if n_a > 0:
                    total_utility += gamma_O**t * (np.sum(structure.EAD) + grd.vacancy_factor(n_a, n_d) * structure.community_cost)
                else:
                    total_utility += gamma_O**t * np.sum(structure.EAD)
        return total_utility

    def objective_function(self, strategies, structures, gamma_O, grd):
        """
        Implements Oracle's objective function.
        
        \min_{\pi_n} \sum P(\pi_n) \cdot U_O(\pi_n)
        
        :param strategies: dict - Strategy decision set π_n = {s_k1, s_k2, ..., s_kN}.
        :param structures: list - List of Structure objects.
        :param gamma_O: float - Oracle's discount factor.
        :param grd: GRD object - Used to calculate the vacancy factor.
        :return: float - The value of the objective function.
        """
        total_objective = 0
        for strategy in strategies.values():
            probability = 1 / len(strategies)  # Assume equal probability for simplicity
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
