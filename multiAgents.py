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

import pdb
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        food_max = float('inf')
        ghost_max = float('inf')
        x0, y0 = newPos

        score = successorGameState.getScore()
        if len(newFood.asList()) > 0:
            for food in newFood.asList():
                distance = manhattanDistance(newPos, food)
                if distance < food_max and distance > 0:
                    food_max = distance
        else:
            food_max = 0
        if len(newGhostStates) > 0:   
            for x in newGhostStates:
                distance_ghost = manhattanDistance(newPos, x.getPosition())
                ghost_max = min(ghost_max, distance_ghost)                

        ghost_max += 1
        food_max = food_max/3
        return score - 9/ghost_max - food_max

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        return self.minimax(gameState, self.depth + 1, agent_index=self.index)[1]
    def minimax(self, gameState, depth, agent_index):
        if agent_index == 0:
            depth -= 1
        if depth == 0 or gameState.isWin() or gameState.isLose(): #or (agent_index == gameState.getNumAgents() - 1 and depth == 1) :
            return self.evaluationFunction(gameState), Directions.STOP
        if agent_index == 0:
            # depth -= 1
            return self.maximizing(gameState, depth, agent_index)
        else:
            # print(" "*depth, agent_index)
            return self.minimizing(gameState, depth, agent_index)
        
    def minimizing(self, gameState, depth, agent_index):
        minimum_score = float("inf")
        minimum_action = Directions.STOP
        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)
            agent_index_next = (agent_index + 1) % gameState.getNumAgents()
            temp = self.minimax(successor_state, depth, agent_index_next)[0]
            # pdb.set_trace()
            if temp < minimum_score:
                minimum_score = temp
                minimum_action = action
        return minimum_score, minimum_action

    def maximizing(self, gameState, depth, agent_index):
        maximum_score = float("-inf")
        maximum_action = Directions.STOP
        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)
            agent_index_next = (agent_index + 1) % gameState.getNumAgents()
            temp = self.minimax(successor_state, depth, agent_index_next)[0]
            if temp > maximum_score:
                maximum_score = temp
                maximum_action = action
        return maximum_score, maximum_action




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        minimum = float("-inf")
        maximum = float("inf")
        return self.pruning(gameState, self.depth + 1, self.index, minimum, maximum)[1]

    def pruning(self, gameState, depth, agent_index, alpha, beta):
        if agent_index == 0:
           depth -= 1
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agent_index == 0:
            return self.alpha_pruning(gameState, depth, agent_index, alpha, beta)
        else:
            return self.beta_pruning(gameState, depth, agent_index, alpha, beta)
        
    def alpha_pruning(self, gameState, depth, agent_index, alpha, beta):
        maximum_score = float("-inf")
        maximum_action = Directions.STOP
        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)
            agent_index_next = (agent_index + 1) % gameState.getNumAgents()
            temp = self.pruning(successor_state, depth, agent_index_next, alpha, beta)[0]
            # pdb.set_trace()
            if temp > maximum_score:
                maximum_score = temp
                maximum_action = action
            if temp > beta:
                return temp, action
            if maximum_score > alpha:
                alpha = maximum_score
        return maximum_score, maximum_action

    def beta_pruning(self, gameState, depth, agent_index, alpha, beta):
        minimum_score = float("inf")
        minimum_action = Directions.STOP
        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)
            agent_index_next = (agent_index + 1) % gameState.getNumAgents()
            temp = self.pruning(gameState.generateSuccessor(agent_index, action), depth, agent_index_next, alpha, beta)[0]
            # pdb.set_trace()
            if temp < minimum_score:
                minimum_score = temp
                minimum_action = action
            if temp < alpha:
                return temp, action
            if minimum_score < beta:
                beta = minimum_score
        return minimum_score, minimum_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, self.depth + 1, agent_index = self.index)[1]
    def expectimax(self, gameState, depth, agent_index):
        # pdb.set_trace
        if agent_index == 0:
            depth -= 1
        if depth == 0 or gameState.isWin() or gameState.isLose(): 
            return self.evaluationFunction(gameState), Directions.STOP
        if agent_index == 0:
            return self.maximizing(gameState, depth, agent_index)
        else:
            return self.average(gameState, depth, agent_index)
        
    def average(self, gameState, depth, agent_index):
        average_score = 0
        count = 0
        minimum_action = Directions.STOP
        for action in gameState.getLegalActions(agent_index):
            # pdb.set_trace()
            successor_state = gameState.generateSuccessor(agent_index, action)
            agent_index_next = (agent_index + 1) % gameState.getNumAgents()
            temp = self.expectimax(successor_state, depth, agent_index_next)[0]
            count += 1
            average_score += temp
            maximum_action = action
        return average_score/count, minimum_action

    def maximizing(self, gameState, depth, agent_index):
        maximum_score = float("-inf")
        maximum_action = Directions.STOP
        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)
            agent_index_next = (agent_index + 1) % gameState.getNumAgents()
            temp = self.expectimax(successor_state, depth, agent_index_next)[0]
            # pdb.set_trace()
            if temp > maximum_score:
                maximum_score = temp
                maximum_action = action
        return maximum_score, maximum_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # pdb.set_trace()
    score = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghost_max = float("inf")
    food_min = float("inf")

    Scared_factor = 5
    Food_factor = 1
    Ghost_factor = -1

    food_min_list = []
    if len(newFood.asList()) > 0:
        for food in newFood.asList():
            # pdb.set_trace()
            food_min_list.append(manhattanDistance(newPos, food))
        food_min = min(food_min_list)
        score += Food_factor/food_min
    else:
        food_min = 0


    if len(newGhostStates) > 0:   
        for ghost in newGhostStates:
            distance_ghost = manhattanDistance(newPos, ghost.getPosition())
            scared_time = ghost.scaredTimer

            if distance_ghost > 0:
                if scared_time == 0:
                    score += Ghost_factor/distance_ghost  
                else:
                    score += Scared_factor/distance_ghost
                            
    return score


# Abbreviation
better = betterEvaluationFunction
