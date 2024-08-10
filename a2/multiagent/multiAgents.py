# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

from functools import partial
from math import inf, log

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        # Calculate the distance from the ghost to Pacman
        distance_to_pacman = partial(manhattanDistance, newPos)

        # Function to evaluate the ghost's impact on Pacman
        def evaluate_ghost(ghost):
            distance = distance_to_pacman(ghost.getPosition())
            
            if ghost.scaredTimer > distance:
                return float("inf")
            if distance <= 1:
                return float("-inf")
            
            return 0

        # Calculate the minimum impact of the ghosts on Pacman
        ghost_impact = min(map(evaluate_ghost, newGhostStates))

        # Calculate the distance to the closest food
        distances_to_food = map(distance_to_pacman, newFood.asList())
        closest_food_distance = min(distances_to_food, default=float("inf"))

        # Calculate the feature for the closest food
        closest_food_feature = 1.0 / (1.0 + closest_food_distance)

        # Combine the scores for Pacman's actions
        evaluated_score = successorGameState.getScore() + ghost_impact + closest_food_feature

        # Return the evaluated score
        return evaluated_score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        def minimax(state, depth, current_agent):
            
            "Returns the best value-action pair for the current agent"
            
            # Calculate the depth for the next step
            next_depth = depth - 1 if current_agent == 0 else depth
            
            # Check if the game is over or if the maximum depth is reached
            if next_depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            # Define the best function and best value depending on the current agent
            best_function, best_value = (max, float("-inf")) if current_agent == 0 else (min, float("inf"))
            next_agent = (current_agent + 1) % state.getNumAgents()
            best_action = None
            
            # Loop through legal actions
            for action in state.getLegalActions(current_agent):
                successor_state = state.generateSuccessor(current_agent, action)
                value_of_action, _ = minimax(successor_state, next_depth, next_agent)
                
                # Update the best value and action based on the best function
                if best_function(best_value, value_of_action) == value_of_action:
                    best_value = value_of_action
                    best_action = action

            return best_value, best_action

        # Call the minimax function to find the best action for the current agent
        value, best_action = minimax(gameState, self.depth + 1, self.index)

        # Return the best action
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def alpha_beta(state, depth, alpha, beta, current_agent):
            # Check if the current agent is a maximizing agent (Pacman)
            is_max = current_agent == 0
            
            # Calculate the depth for the next step
            next_depth = depth - 1 if is_max else depth
            
            # Check if we've reached the maximum depth or the game is over
            if next_depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            # Determine the index of the next agent
            next_agent = (current_agent + 1) % state.getNumAgents()
            
            # Initialize the best value and best action based on whether the current agent is a maximizing agent
            best_value = float("-inf") if is_max else float("inf")
            best_action = None
            
            # Determine the best function (max or min) based on whether the current agent is maximizing
            best_function = max if is_max else min
            
            # Loop through legal actions
            for action in state.getLegalActions(current_agent):
                successor_state = state.generateSuccessor(current_agent, action)
                
                # Recursively call alpha-beta for the next agent
                value_of_action, _ = alpha_beta(
                    successor_state, next_depth, alpha, beta, next_agent)
                
                # Update the best value and action based on the best function
                if best_function(best_value, value_of_action) == value_of_action:
                    best_value = value_of_action
                    best_action = action

                if is_max:
                    # Check if the best value is greater than beta
                    if best_value > beta:
                        return best_value, best_action
                    alpha = max(alpha, best_value)
                else:
                    # Check if the best value is less than alpha
                    if best_value < alpha:
                        return best_value, best_action
                    beta = min(beta, best_value)

            return best_value, best_action

        # Call the alpha_beta function to find the best action for the current agent
        _, best_action = alpha_beta(gameState, self.depth + 1, float("-inf"), float("inf"), self.index)

        # Return the best action
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Determine the agent index
        agent = self.index

        # If the agent is not the maximizing player, choose a random action
        if agent != 0:
            return random.choice(state.getLegalActions(agent))

        def expectimax(state, depth, agent):
            # Check if we've reached the specified depth or the game has ended
            nextDepth = depth - 1 if agent == 0 else depth
            if nextDepth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            nextAgent = (agent + 1) % state.getNumAgents()
            legalMoves = state.getLegalActions(agent)

            # If it's not the maximizing player's turn, calculate expected value
            if agent != 0:
                prob = 1.0 / float(len(legalMoves))
                value = 0.0
                for action in legalMoves:
                    successorState = state.generateSuccessor(agent, action)
                    exp_value, _ = expectimax(successorState, nextDepth, nextAgent)
                    value += prob * exp_value
                return value, None

            # If it's the maximizing player's turn, choose the action with the maximum expected value
            best_value, bestAction = float("-inf"), None
            for action in legalMoves:
                successorState = state.generateSuccessor(agent, action)
                exp_value, _ = expectimax(successorState, nextDepth, nextAgent)
                if max(best_value, exp_value) == exp_value:
                    best_value, bestAction = exp_value, action
            return best_value, bestAction

        # Determine the best action for the maximizing player using the Expectimax algorithm
        _, action = expectimax(gameState, self.depth + 1, self.index)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: It calculates a score for a given game state based on various factors. If the game state represents a win, it returns positive infinity, and if it represents a loss, it returns negative infinity. Otherwise, it considers the current score, the proximity to the nearest food pellets, and the distance to the nearest ghosts. The function also allows you to customize the weight of these factors to influence Pacman's decision-making process and behavior in the game. It's a vital component for evaluating and improving the performance of a Pacman agent.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("infinity")
    if currentGameState.isLose():
        return -float("infinity")

    totalScore = currentGameState.getScore()
    currentPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostPositions = [currentGameState.getGhostPosition(i) for i in range(1, currentGameState.getNumAgents())]

    # Calculate distance to the nearest food
    minFoodDistance = min(manhattanDistance(currentPos, food) for food in foodList)

    # Calculate distance to the nearest ghost
    minGhostDistance = min(manhattanDistance(currentPos, ghost) for ghost in ghostPositions)

    # Heuristics
    foodWeight = 2.0
    ghostWeight = 4.0
    scoreWeight = 1.0

    totalScore += scoreWeight * currentGameState.getScore()
    totalScore -= foodWeight * minFoodDistance
    totalScore -= ghostWeight * minGhostDistance

    return totalScore

# Abbreviation
better = betterEvaluationFunction

# reference: https://github.com/DylanCope/CS188-Multi-Agent/blob/master/multiAgents.py
# betterEvaluationFunction optimized through ChatGPT.
