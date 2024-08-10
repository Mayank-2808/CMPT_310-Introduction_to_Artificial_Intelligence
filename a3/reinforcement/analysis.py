# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():

# We aim to limit the agent's lifespan to only 3 steps, hence a significant negative living penalty.
# There should be no fear of fire, so no noise is considered.
# The agent should move towards a small reward, so we discount the value of a reward by 10.

    answerDiscount = 0.1
    answerNoise = 0
    answerLivingReward = -4.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():

# We aim for the agent to survive for 7 steps, emphasizing longevity.
# We introduce some noise to make the agent cautious around fire.

    answerDiscount = 0.1
    answerNoise = 0.1
    answerLivingReward = -1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():

# We aim for the agent's lifespan to be 5 steps, focusing on medium-term survival.
# To eliminate any fear of fire, we introduce no noise in the environment.

    answerDiscount = 1.0
    answerNoise = 0.0
    answerLivingReward = -1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():

# To prevent falling off cliffs, we introduce some noise to make the agent cautious.
# Emphasizing a distant reward of 10 means using a discount factor (alpha) of 1.
# Encourage the agent to live for at least 10 steps by applying a small living penalty.

    answerDiscount = 1.0
    answerNoise = 0.1
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():

# We prioritize the agent's eternal survival with a substantial living reward.
# To ensure the agent avoids cliffs, we introduce noise for cautious navigation.
# As long as the living reward exceeds 10, there's no need for additional discounting.

    answerDiscount = 1.0
    answerNoise = 0.1
    answerLivingReward = 100.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    # return answerEpsilon, answerLearningRate
    # If not possible,
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))

# reference: https://github.com/battmuck32138/reinforcement_learning/blob/master/analysis.py
