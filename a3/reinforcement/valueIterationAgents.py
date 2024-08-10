# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        # Execute 'k' iterations for value function estimation
        for _ in range(self.iterations):
            
            # Create a copy of the current value function to store updated values
            updated_value_function = self.values.copy()

            # Compute Q-values for each possible next state ('s_prime')
            for state in self.mdp.getStates():
                q_values = [float('-inf')]  # Initialize a list to store Q-values with negative infinity
                is_terminal_state = self.mdp.isTerminal(state)  # Determine if the state is terminal (boolean)

                # Terminal states have an assigned value of 0
                if is_terminal_state:
                    updated_value_function[state] = 0
                else:
                    # Obtain a list of permissible actions in the current state
                    legal_actions = self.mdp.getPossibleActions(state)

                    # Calculate Q-values for each legal action
                    for action in legal_actions:
                        q_values.append(self.getQValue(state, action))

                    # Update the value function for state 's' with the maximum Q-value
                    updated_value_function[state] = max(q_values)

            # Replace the current value function with the updated one
            self.values = updated_value_function



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        # Calculate the expected value for taking 'action' in 'state'
        possible_next_states = self.mdp.getTransitionStatesAndProbs(state, action)  # List of (next_state, transition_prob) pairs
        expected_value = []

        for next_state, transition_prob in possible_next_states:
            reward = self.mdp.getReward(state, action, next_state)
            discounted_future_value = self.discount * self.values[next_state]
            expected_value.append(transition_prob * (reward + discounted_future_value))

        return sum(expected_value)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        # Determine the best legal action to take in the current 'state'
        legal_actions = self.mdp.getPossibleActions(state)

        # If there are no legal actions, return None
        if len(legal_actions) == 0:
            return None

        # Compute Q-values for each legal action and store them in a list
        action_q_value_pairs = []  # List of (action, Q-value) pairs

        for action in legal_actions:
            q_value = self.getQValue(state, action)
            action_q_value_pairs.append((action, q_value))

        # Identify the action with the highest Q-value
        best_action_with_max_q_value = max(action_q_value_pairs, key=lambda x: x[1])[0]

        # Return the best action based on the highest Q-value
        return best_action_with_max_q_value

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        """
        Perform cyclic value iteration over the states of the MDP.

        Each state's value is updated in a cyclic manner. If the state is terminal, no update is performed.
        """
        states = self.mdp.getStates()

        # Initialize the value function with all states having a value of 0.
        for state in states:
            self.values[state] = 0

        num_states = len(states)

        for i in range(self.iterations):
            state_index = i % num_states
            state = states[state_index]
            is_terminal = self.mdp.isTerminal(state)

            if not is_terminal:
                action = self.getAction(state)
                q_value = self.getQValue(state, action)
                self.values[state] = q_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        """
        Perform value iteration to estimate the optimal value function.

        We use a priority queue to prioritize state updates based on the maximum Q-value change.
        Predecessors are tracked to assist in state updates.
        """

        # Get the list of all states in the Markov Decision Process
        states = self.mdp.getStates()

        # Initialize a priority queue to prioritize state updates
        state_priority_queue = util.PriorityQueue()

        # Maintain a dictionary to keep track of predecessors
        predecessors = {}

        for state in states:
            self.values[state] = 0
            predecessors[state] = self.findPredecessors(state)

        for state in states:
            is_terminal = self.mdp.isTerminal(state)

            if not is_terminal:
                current_value_of_state = self.values[state]
                diff = abs(current_value_of_state - self.maxQValue(state))
                state_priority_queue.push(state, -diff)

        for _ in range(self.iterations):

            if state_priority_queue.isEmpty():
                return

            current_state = state_priority_queue.pop()
            self.values[current_state] = self.maxQValue(current_state)

            for predecessor in predecessors[current_state]:
                diff = abs(self.values[predecessor] - self.maxQValue(predecessor))
                if diff > self.theta:
                    state_priority_queue.update(predecessor, -diff)

    def maxQValue(self, state):
        return max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])

    def findPredecessors(self, state):
        """
        Find predecessors of a state by examining all other states and their actions.

        Predecessors are states that have a nonzero probability of reaching the given state
        by taking some action (T > 0), excluding terminal states.
        """
        predecessor_set = set()
        all_states = self.mdp.getStates()
        movements = ['north', 'south', 'east', 'west']

        if not self.mdp.isTerminal(state):
            for other_state in all_states:
                is_terminal = self.mdp.isTerminal(other_state)
                legal_actions = self.mdp.getPossibleActions(other_state)

                if not is_terminal:
                    for move in movements:
                        if move in legal_actions:
                            transitions = self.mdp.getTransitionStatesAndProbs(other_state, move)

                            for next_state, transition_prob in transitions:
                                if next_state == state and transition_prob > 0:
                                    predecessor_set.add(other_state)

        return predecessor_set

#reference: https://github.com/battmuck32138/reinforcement_learning/blob/master/analysis.py
