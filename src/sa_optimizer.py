import numpy as np

def simulated_annealing(fitness_func, n_features, initial_temp=100.0, 
                       final_temp=0.01, cooling_rate=0.95, iterations_per_temp=10,
                       random_state=42):
    """
    Simulated Annealing for binary feature selection.
    
    Parameters:
    -----------
    fitness_func : callable
        Function that takes a binary vector and returns fitness (lower is better)
    n_features : int
        Number of features
    initial_temp : float
        Initial temperature
    final_temp : float
        Final temperature
    cooling_rate : float
        Temperature reduction factor
    iterations_per_temp : int
        Number of iterations at each temperature
    random_state : int
        Random seed
    
    Returns:
    --------
    best_solution : numpy array
        Best feature subset found
    best_fitness : float
        Best fitness value
    fitness_history : list
        Fitness history
    """
    np.random.seed(random_state)
    
    # Initialize solution
    current_solution = np.random.randint(0, 2, n_features)
    if np.sum(current_solution) == 0:
        current_solution[np.random.randint(0, n_features)] = 1
    
    current_fitness = fitness_func(current_solution)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    fitness_history = [current_fitness]
    
    temperature = initial_temp
    iteration = 0
    
    while temperature > final_temp:
        for _ in range(iterations_per_temp):
            # Generate neighbor by flipping one random bit
            neighbor = current_solution.copy()
            flip_idx = np.random.randint(0, n_features)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            
            # Ensure at least one feature selected
            if np.sum(neighbor) == 0:
                neighbor[np.random.randint(0, n_features)] = 1
            
            neighbor_fitness = fitness_func(neighbor)
            
            # Accept or reject
            delta = neighbor_fitness - current_fitness
            
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                # Update best
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_solution = current_solution.copy()
            
            fitness_history.append(current_fitness)
            iteration += 1
        
        # Cool down
        temperature *= cooling_rate
    
    return best_solution, best_fitness, fitness_history

