import random
import numpy as np
#from deap import base, creator, tools, algorithms
import pandas
import pandas as pd
import os
import matplotlib.pyplot as plt

def fitness(individual, items, max_weight):
    """
        Function: This function calculates the 
                    fitness score (total value) of an individual solution. If the bit is 1, it adds the corresponding item's weight and 
                    value to the running totals.
        Return:   If the total weight exceeds the maximum weight capacity, it returns 0 (invalid solution). Otherwise, it returns the 
                    total value       
    """
    total_weight = 0
    total_value = 0
    for i, bit in enumerate(individual):
        if bit:
            total_weight += items['weights'][i]
            total_value += items['values'][i]
    if total_weight > max_weight:
        penalty = (total_weight - max_weight) ** 2  # Quadratic penalty
        fitness_score = total_value - penalty
    else:
        fitness_score = total_value
    return fitness_score

def repair_individual(individual, items, max_weight):
    """
    Repairs an individual by removing items until the total weight is within the maximum capacity.
    Items are removed based on the lowest value-to-weight ratio.
    """
    total_weight = sum(individual[i] * items['weights'][i] for i in range(len(individual)))
    if total_weight <= max_weight:
        return individual  # Individual is already feasible
    else:
        # Calculate value-to-weight ratios for items included in the individual
        included_items = [(i, items['values'][i] / items['weights'][i]) for i in range(len(individual)) if individual[i] == 1]
        # Sort items by lowest value-to-weight ratio (least valuable per unit weight)
        included_items.sort(key=lambda x: x[1])
        # Remove items until the total weight is within capacity
        for i, _ in included_items:
            individual[i] = 0  # Remove the item
            total_weight -= items['weights'][i]
            if total_weight <= max_weight:
                break
        return individual

def create_dataset(file_name):
    """
        Reads a dataset from a file and extracts weights, values, maximum capacity, item count, and optimal value.

        Parameters:
        file_name (str): The name of the file containing the dataset.

        Returns:
        tuple: A tuple containing:
            - item_dict (dict): A dictionary with two keys:
                - "weights" (list): A list of weights of the items.
                - "values" (list): A list of values of the items.
            - max_capacity (int): The maximum capacity of the knapsack.
            - item_count (int): The number of items.
            - optimal_value (int): The optimal value for the given maximum capacity.
    """
    weights = []
    values = []
    max_capacity = 0   # value of maximume weights
    item_count = 0     # number of the items for each individual
    optimal_value = 0  

    # Read file and extract data file
    full_path = os.path.abspath(__file__) # Get the full path of the script     
    script_directory = os.path.dirname(full_path) # Get the directory of the script
    data_file = os.path.join(script_directory,file_name) # Get the full path of the data file

    with open(data_file,'r') as file: 
        data = file.readlines()      

        for idx, line in enumerate(data): # extract weights and vlues and store it into list
            x = line.split()
            if idx == 0:
                max_capacity = int(x[1])
                item_count = int(x[0])
            else:
                weights.append(int(x[1]))
                values.append(int(x[0]))
        
        # Find the vlaue of optimal_value paramener. depend on value of (max_capacity) 
        if max_capacity == 269: optimal_value = 295
        elif max_capacity == 10000: optimal_value = 9767
        else: optimal_value = 1514
        
        item_dict = {"weights":weights ,"values":values}

    return item_dict, max_capacity, item_count, optimal_value

def select_best_individuals(population, fitness, num_selected):
    selected_indices = np.argsort(fitness)[-num_selected:]
    return population[selected_indices]

def calculate_weighted_probability(ind_fitness, individuals): #, values):
    """
    Calculate the weighted probability for a given set of weights and values.
    
    Parameters:
    - weights: A list of weights for each event.
    - values: A list of values for each event corresponding to the weights.
    
    Returns:
    - A list of weighted probabilities for each event.
    """
    
    # Step 1: Normalize the fitness values to get weights that sum to 1
    total_fitness = sum(f for f in ind_fitness if f > 0)
    normalized_weights = [(f / total_fitness if f > 0 else 0) for f in ind_fitness]
    
    n_bits = len(individuals[0])  # Number of bits per individual
    weighted_probabilities = []

    # Calculate the probability for each bit position
    for bit_position in range(n_bits):
        weighted_sum_for_bit_1 = 0.0
        for i, individual in enumerate(individuals):
            if individual[bit_position] == 1:
                weighted_sum_for_bit_1 += normalized_weights[i]
        
        weighted_probabilities.append(weighted_sum_for_bit_1)
    
    return weighted_probabilities

def plot_curve(best_values):
    """
    Plot the average fitness of the best solution weights over generations.
    
    Parameters:
    - best_weights: A list of the best solution weights for each generation.
    """
    average_fitness = []

    #for weights in best_weights:
    #    average_fitness.append(sum(weights) / len(best_weights[0]))
    #average_fitness.append(sum(best_weights) / len(best_weights))

    average_fitness = np.mean(best_values, axis=0)

    generations = list(range(1, len(average_fitness) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, average_fitness, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Generations')
    plt.ylabel('Average Fitness of 5 Best Solution Weights')
    plt.title('Average Fitness Over Generations')
    plt.grid(True)
    plt.show()

def main():
    population_size = 50   # Population size is the number of individuals in each generation
    num_generations  = 50   # number of generations to run the Genetic Algorithm
    dataset_files = ['23_10000', '10_269','100_995']
    seed_ = [20, 30, 40, 50, 60]

    for dataset_file in dataset_files:
        print("=====================================")
        print(f"Dataset: {dataset_file}")

        # Create a dataset
        item_dict, max_capacity, item_count, optimal_value = create_dataset(dataset_file)
        print(f"Optimal value: {optimal_value}")
        print()

        # Reset the best_individual, best_weight, and best_value lists
        seletion_size = round(population_size / 2) # The number of individuals to select from intire population, and use them to create offspring
        elitism_size = 2 # The number of individuals to select from intire population, and use them to create offspring
        runs = 5
        li_best_values_run = [] # list of best values for each run
    
        for run in range(runs):
            best_value = None
            best_individual = None
            
            li_best_individuals_gen = [] # list of best individuals for each generation
            li_best_values_gen = []
            random.seed(seed_[run])  # random seed
            # INITIALIZATION: Initialize random individual. then add it to population
            individuals = np.random.randint(2, size=(population_size, item_count))  # It generates a random integer between 0 and 1 (inclusive)
            population = individuals

            # Repair initial population
            for i in range(population_size):
                population[i] = repair_individual(population[i], item_dict, max_capacity)

            for generation in range(num_generations):
                # EVALUATE FITNESS: Calculate the fitness of the individual
                fitness_score = [fitness(individual, item_dict, max_capacity) for individual in population]
                
                # Elitsm: Select two best individual from the current population
                elitsm_individuals = sorted(population, key=lambda x: fitness(x, item_dict, max_capacity), reverse=True)[:elitism_size]

                # SELECTION: Sort and Select a subset of the best-performing candidates from the current population based on fitness
                selected_individuals = select_best_individuals(population, fitness_score, seletion_size)
 
                # MARGINAL DISTRIBUTION ESTIMATION: Estimate the probability of each bit being 1 in the selected individuals
                # calcualte the probability of each bit being 1 in the selected individuals
                # probability = np.mean(selected_individuals, axis=0)
                probability = calculate_weighted_probability([fitness(individual, item_dict, max_capacity) for individual in selected_individuals], selected_individuals)

                # initialize zero vector for (SAMPLE NEW POPULATION): 
                new_population = np.zeros((population_size, len(probability)), dtype=int)
                
                # Iterate over the new population and samble new population, 
                # Generate a new population by sampling from the estimated probability distribution
                for i in range(len(new_population) - elitism_size):
                    new_individual = (np.random.rand(len(probability)) < probability).astype(int)
                    new_population[i] = new_individual

                # Add the elitsm_individuals to the new_population
                new_population[:elitism_size] = elitsm_individuals

                # Update the population
                population = new_population

                # TRACK BEST INIVIDUAL: Track the best individual, weight, and value
                current_best_individual = max(population, key=lambda x: fitness(x, item_dict, max_capacity))
                current_best_value = fitness(current_best_individual, item_dict, max_capacity)
                # Store the best individual and value for each generation
                li_best_individuals_gen.append(current_best_individual)
                li_best_values_gen.append(current_best_value)
                
                if best_value is None or current_best_value > best_value:  # Check if the current individual is better than the previous best individual
                    best_individual = current_best_individual
                    #best_weight = current_best_weight
                    best_value = current_best_value

                
                # Stoping criteria
                #if best_value == optimal_value:
                #    break
            
            # Store the best value for each run
            li_best_values_run.append(li_best_values_gen)


            # Print the results
            print(f"RUN: {run + 1}")
            print(f"Best Individual: {best_individual}")
            print(f"Best Value: {best_value}, at generation {generation + 1}")
            print()
            #print(f"Average Best Value: {avg_best_value}")
            #print(f"Standard Deviation of Best Value: {std_best_value}")
            #print()
        
        # Clacular the average value of the best solution weights over generations
        best_value_each_run = [max(a) for a in li_best_values_run]
        avg_best_value = sum(best_value_each_run) / runs
        std_best_value = np.std(best_value_each_run, ddof=1)
        print(f"Average Best Value: {avg_best_value}")
        print(f"Standard Deviation of Best Value: {std_best_value}")
        print()

        # Plot the average fitness of the best solution weights over generations
        plot_curve(li_best_values_run)
                


if __name__ == "__main__":
    main()