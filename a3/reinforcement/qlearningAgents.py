# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Get the list of legal actions for the given state
        legal_actions = self.getLegalActions(state)

        # Check if there are legal actions available
        if legal_actions:
            # If there are legal actions, calculate and return the maximum Q-value
            max_q_value = max([self.getQValue(state, action) for action in legal_actions])
            return max_q_value
        else:
            # If there are no legal actions, return a default value of 0.0
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get the list of legal actions for the given state
        legal_actions = self.getLegalActions(state)

        # Check if there are legal actions available
        if not legal_actions:
            # If there are no legal actions, return None
            return None

        # Create a list of action-QValue pairs for each legal action
        action_QValue_pairs = [(action, self.getQValue(state, action)) for action in legal_actions]

        # Find the pair with the maximum Q-value using a lambda function as the key
        max_pair = max(action_QValue_pairs, key=lambda x: x[1])

        # Return the action from the pair with the maximum Q-value
        return max_pair[0]


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_Actions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        
        # Randomly decide whether to explore or exploit
        random_action = util.flipCoin(self.epsilon)

        # Check if there are legal actions available
        if not legal_Actions:
            # If there are no legal actions, return None
            return action

        if random_action:
            # If the random action is selected, choose a random action from the legal actions
            action = random.choice(legal_Actions)
            return action
        else:
            # If not exploring randomly, use the policy to choose the action
            action = self.getPolicy(state)
            return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Get the reward R(s, a, s')
        R = reward

        # Get the maximum Q-value for the next state Q(s', a')
        max_Q = self.getValue(nextState)

        # Get the discount factor gamma
        gamma = self.discount

        # Get the current Q-value Q(s, a)
        Q = self.qvalues[(state, action)]

        # Calculate the sample value: sample = R(s, a, s') + gamma * max_a' Q(s', a')
        sample = R + gamma * max_Q

        # Update the Q-value: Q(s, a) = (1 - alpha) * Q(s, a) + alpha * sample
        self.qvalues[(state, action)] = (1 - self.alpha) * Q + self.alpha * sample


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Import the NumPy library
        import numpy as np

        # Get the weight vector w as a NumPy array
        w = np.array(self.getWeights())

        # Get the feature vector as a NumPy array using the feature extractor
        feature_vector = np.array(self.featExtractor.getFeatures(state, action))

        # Calculate the Q-value as the dot product of the weight vector and feature vector: Q(s, a) = np.dot(w, feature_vector)
        return np.dot(w, feature_vector)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Calculate the difference between the target Q-value and the current Q-value
        # difference = (r + gamma * max_a' Q(s', a')) - Q(s, a)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)

        # Get the feature vector using the feature extractor
        feature_vector = self.featExtractor.getFeatures(state, action)

        # Update the weight vector using the TD error
        for feature in feature_vector:
            # Update the weight for each feature using the learning rate (alpha) and TD error
            self.weights[feature] += self.alpha * difference * feature_vector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

#reference: https://github.com/battmuck32138/reinforcement_learning/blob/master/analysis.py
