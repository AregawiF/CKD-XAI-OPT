import numpy as np

def binary_pso(fitness_func, n_features, n_particles=30, n_iterations=100,
               w=0.7, c1=1.5, c2=1.5, random_state=42):
    """
    Binary Particle Swarm Optimization for feature selection.
    
    Parameters:
    -----------
    fitness_func : callable
        Function that takes a binary vector and returns fitness (lower is better)
    n_features : int
        Number of features
    n_particles : int
        Number of particles
    n_iterations : int
        Number of iterations
    w : float
        Inertia weight
    c1 : float
        Cognitive parameter
    c2 : float
        Social parameter
    random_state : int
        Random seed
    
    Returns:
    --------
    best_solution : numpy array
        Best feature subset found
    best_fitness : float
        Best fitness value
    fitness_history : list
        Best fitness per iteration
    """
    np.random.seed(random_state)
    
    # Initialize particles
    positions = []
    velocities = np.random.uniform(-4, 4, (n_particles, n_features))
    
    for _ in range(n_particles):
        pos = np.random.randint(0, 2, n_features)
        if np.sum(pos) == 0:
            pos[np.random.randint(0, n_features)] = 1
        positions.append(pos)
    
    positions = np.array(positions)
    
    # Initialize personal bests
    personal_best_positions = positions.copy()
    personal_best_fitness = np.array([fitness_func(pos) for pos in positions])
    
    # Initialize global best
    global_best_idx = np.argmin(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_fitness = personal_best_fitness[global_best_idx]
    fitness_history = [global_best_fitness]
    
    # PSO main loop
    for iteration in range(n_iterations):
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_best_positions[i] - positions[i]) +
                           c2 * r2 * (global_best_position - positions[i]))
            
            # Clamp velocity
            velocities[i] = np.clip(velocities[i], -4, 4)
            
            # Update position using sigmoid function
            sigmoid = 1 / (1 + np.exp(-velocities[i]))
            positions[i] = (np.random.rand(n_features) < sigmoid).astype(int)
            
            # Ensure at least one feature selected
            if np.sum(positions[i]) == 0:
                positions[i, np.random.randint(0, n_features)] = 1
            
            # Evaluate fitness
            current_fitness = fitness_func(positions[i])
            
            # Update personal best
            if current_fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = current_fitness
                personal_best_positions[i] = positions[i].copy()
                
                # Update global best
                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = positions[i].copy()
        
        fitness_history.append(global_best_fitness)
    
    return global_best_position, global_best_fitness, fitness_history

