import numpy as np
import random

def genetic_algorithm(fitness_func, n_features, n_population=50, n_generations=100, 
                     crossover_rate=0.8, mutation_rate=0.1, random_state=42):
    """
    Genetic Algorithm for binary feature selection.
    
    Parameters:
    -----------
    fitness_func : callable
        Function that takes a binary vector and returns fitness (lower is better)
    n_features : int
        Number of features
    n_population : int
        Population size
    n_generations : int
        Number of generations
    crossover_rate : float
        Probability of crossover
    mutation_rate : float
        Probability of mutation per bit
    random_state : int
        Random seed
    
    Returns:
    --------
    best_solution : numpy array
        Best feature subset found
    best_fitness : float
        Best fitness value
    fitness_history : list
        Best fitness per generation
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Initialize population
    population = []
    for _ in range(n_population):
        individual = np.random.randint(0, 2, n_features)
        # Ensure at least one feature is selected
        if np.sum(individual) == 0:
            individual[np.random.randint(0, n_features)] = 1
        population.append(individual)
    
    population = np.array(population)
    
    # Evaluate initial population
    fitness_scores = np.array([fitness_func(ind) for ind in population])
    best_idx = np.argmin(fitness_scores)
    best_solution = population[best_idx].copy()
    best_fitness = fitness_scores[best_idx]
    fitness_history = [best_fitness]
    
    # Evolution loop
    for generation in range(n_generations):
        # Selection (tournament selection)
        new_population = []
        for _ in range(n_population):
            # Tournament of size 2
            idx1, idx2 = np.random.choice(n_population, 2, replace=False)
            winner = idx1 if fitness_scores[idx1] < fitness_scores[idx2] else idx2
            new_population.append(population[winner].copy())
        
        new_population = np.array(new_population)
        
        # Crossover
        for i in range(0, n_population - 1, 2):
            if np.random.rand() < crossover_rate:
                # Single-point crossover
                crossover_point = np.random.randint(1, n_features)
                temp = new_population[i, crossover_point:].copy()
                new_population[i, crossover_point:] = new_population[i+1, crossover_point:]
                new_population[i+1, crossover_point:] = temp
                
                # Ensure at least one feature selected
                if np.sum(new_population[i]) == 0:
                    new_population[i, np.random.randint(0, n_features)] = 1
                if np.sum(new_population[i+1]) == 0:
                    new_population[i+1, np.random.randint(0, n_features)] = 1
        
        # Mutation
        for i in range(n_population):
            for j in range(n_features):
                if np.random.rand() < mutation_rate:
                    new_population[i, j] = 1 - new_population[i, j]
            
            # Ensure at least one feature selected
            if np.sum(new_population[i]) == 0:
                new_population[i, np.random.randint(0, n_features)] = 1
        
        population = new_population
        
        # Evaluate new population
        fitness_scores = np.array([fitness_func(ind) for ind in population])
        current_best_idx = np.argmin(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        
        # Update global best
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx].copy()
        
        fitness_history.append(best_fitness)
    
    return best_solution, best_fitness, fitness_history

