# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Initialize the starting node.
    startingNode = problem.getStartState()

    # Check if the starting node is already the goal state.
    if problem.isGoalState(startingNode):
        return []

    # Initialize the stack for DFS.
    stack = util.Stack()

    # Create a list to keep track of visited nodes.
    visitedNodes = []

    # Push the starting node and an empty action list to the stack.
    stack.push((startingNode, []))

    # Start DFS traversal.
    while not stack.isEmpty():
        # Get the current node and the actions taken to reach it.
        currentNode, actions = stack.pop()

        # Check if the current node has not been visited yet.
        if currentNode not in visitedNodes:
            # Mark the current node as visited.
            visitedNodes.append(currentNode)

            # Check if the current node is the goal state.
            if problem.isGoalState(currentNode):
                return actions

            # Explore the successors of the current node.
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                # Create a new list of actions by adding the current action.
                nextAction = actions + [action]
                # Push the next node and its new action list to the stack.
                stack.push((nextNode, nextAction))

    # If no solution is found, return an empty list.
    return []
    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Initialize the starting node.
    startingNode = problem.getStartState()

    # Check if the starting node is already the goal state.
    if problem.isGoalState(startingNode):
        return []

    # Initialize the queue for BFS.
    queue = util.Queue()
    
    # Create a list to keep track of visited nodes.
    visitedNodes = []
    
    # Push the starting node and an empty action list to the queue.
    queue.push((startingNode, []))

    # Start BFS traversal.
    while not queue.isEmpty():
        # Get the current node and the actions taken to reach it.
        currentNode, actions = queue.pop()

        # Check if the current node has not been visited yet.
        if currentNode not in visitedNodes:
            # Mark the current node as visited.
            visitedNodes.append(currentNode)

            # Check if the current node is the goal state.
            if problem.isGoalState(currentNode):
                return actions

            # Explore the successors of the current node.
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                # Create a new list of actions by adding the current action.
                nextAction = actions + [action]
                # Push the next node and its new action list to the queue.
                queue.push((nextNode, nextAction))

    # If no solution is found, return an empty list.
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
   # Initialize the starting node.
    startingNode = problem.getStartState()

    # Check if the starting node is already the goal state.
    if problem.isGoalState(startingNode):
        return []

    # Create a list to keep track of visited nodes.
    visitedNodes = []

    # Initialize the priority queue for uniform cost search.
    priorityQueue = util.PriorityQueue()

    # Push the starting node, an empty action list, and cost 0 to the priority queue.
    priorityQueue.push((startingNode, [], 0), 0)

    # Start uniform cost search.
    while not priorityQueue.isEmpty():
        # Get the current node, actions taken to reach it, and the cost.
        currentNode, actions, oldCost = priorityQueue.pop()

        # Check if the current node has not been visited yet.
        if currentNode not in visitedNodes:
            # Mark the current node as visited.
            visitedNodes.append(currentNode)

            # Check if the current node is the goal state.
            if problem.isGoalState(currentNode):
                return actions

            # Explore the successors of the current node.
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                # Create a new list of actions by adding the current action.
                nextAction = actions + [action]
                # Calculate the new priority based on the old cost and action cost.
                priority = oldCost + cost
                # Push the next node, its new action list, and priority to the priority queue.
                priorityQueue.push((nextNode, nextAction, priority), priority)

    # If no solution is found, return an empty list.
    return []
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize the starting node.
    startingNode = problem.getStartState()

    # Check if the starting node is already the goal state.
    if problem.isGoalState(startingNode):
        return []

    # Create a list to keep track of visited nodes.
    visitedNodes = []

    # Initialize the priority queue for A* search.
    priorityQueue = util.PriorityQueue()

    # Push the starting node, an empty action list, and cost 0 to the priority queue.
    priorityQueue.push((startingNode, [], 0), 0)

    # Start A* search.
    while not priorityQueue.isEmpty():
        # Get the current node, actions taken to reach it, and the old cost.
        currentNode, actions, oldCost = priorityQueue.pop()

        # Check if the current node has not been visited yet.
        if currentNode not in visitedNodes:
            # Mark the current node as visited.
            visitedNodes.append(currentNode)

            # Check if the current node is the goal state.
            if problem.isGoalState(currentNode):
                return actions

            # Explore the successors of the current node.
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                # Create a new list of actions by adding the current action.
                nextAction = actions + [action]
                # Calculate the new cost to the node.
                newCostToNode = oldCost + cost
                # Calculate the heuristic cost using the heuristic function.
                heuristicCost = newCostToNode + heuristic(nextNode, problem)
                # Push the next node, its new action list, and new cost to the priority queue.
                priorityQueue.push((nextNode, nextAction, newCostToNode), heuristicCost)

    # If no solution is found, return an empty list.
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

#reference: https://www.youtube.com/watch?v=%20tx5suCrcYCg
