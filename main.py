# LAB_4_RALENKO
import random
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools

# problem constants:
ONE_MAX_LENGTH = 1000  # length of bit string to be optimized
# Genetic Algorithm constants:
POPULATION_SIZE = 400  # number of individuals in population
P_CROSSOVER = 0.95  # probability for crossover
P_MUTATION = 0.04  # probability for mutating an individual
MAX_GENERATIONS = 55  # max number of generations for stopping condition

RANDOM_SEED = 32
random.seed(RANDOM_SEED)

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint, 0, 1)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# define the population to be a list of individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# the goal ('fitness') function to be maximized
def oneMaxFitness(individual):
    return sum(individual),  # return a tuple


# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", oneMaxFitness)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)


# ----------

def main():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    fitnessValues = list(map(toolbox.evaluate, population))

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual

    print("Start of evolution")

    # Evaluate the entire population

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    print("  Evaluated %i individuals" % len(population))

    # Extracting all the fitnesses of
    fitnessValues = [individual.fitness.values[0] for individual in population]

    # Variable keeping track of the number of generations
    generationCounter = 0

    maxFitnessValues = []
    meanFitnessValues = []

    # Begin the evolution
    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        # A new generation
        generationCounter = generationCounter + 1
        print("-- Generation %i --" % generationCounter)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        print("  Evaluated %i individuals" % len(freshIndividuals))

        # The population is entirely replaced by the offspring
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print("- Generation {}: Max Fitness = {}, Avg Fitness = {}"
              .format(generationCounter, maxFitness, meanFitness))

    print("-- End of (successful) evolution --")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Best Individual = ", *population[best_index], "\n")

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()
