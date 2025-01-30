import random
from deap import base, tools
from deap_individual import create_individual_cb, mutate_individual_cb, crossover_individual_cb, evaluate_individual_cb

class DEAPExecutor:
    """Executor for DEAPTuner."""

    def initialize_toolbox(self):
        """Initialize the DEAP toolbox with operators."""
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_cb)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", crossover_individual_cb)
        toolbox.register("mutate", mutate_individual_cb)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate_individual_cb)
        return toolbox

    def evolve_population(self, toolbox, population, generations, crossover_prob, mutation_prob):
        """Run the evolutionary process."""
        for generation in range(generations):
            for individual in population:
                if (not hasattr(individual.fitness, "values")) or (not individual.fitness.valid):
                    individual.fitness.values = toolbox.evaluate(individual)

            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            population[:] = offspring
            best_individual = tools.selBest(population, 1)[0]

        return tools.selBest(population, 1)[0]
