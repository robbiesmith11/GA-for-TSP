
import pandas as pd
import numpy as np
import random 
import math
import os
os.environ["PYGAME_DETECT_AVX2"] = "1"
import pygame 
import collections
import cProfile
from numba import jit


# References:
#https://stackoverflow.com/questions/16310015/what-does-this-mean-key-lambda-x-x1  used in tournement selection to get the best fitness value as my key to the dictionary is in tuples (identifier, fitness)
# used to sort dictionary and pick elites https://stackoverflow.com/questions/45738414/how-to-install-collections-in-python-3-6 , https://stackoverflow.com/questions/12988351/split-a-dictionary-in-half
# random key value pair from dictionary https://stackoverflow.com/questions/4859292/how-can-i-get-a-random-key-value-pair-from-a-dictionary
# max element of a index of a tuple https://stackoverflow.com/questions/14209062/find-max-of-the-2nd-element-in-a-tuple-python
# learning numba and how to use it to make code faster https://www.youtube.com/watch?v=x58W9A2lnQc&t=445s
# learning numpy vectorization to make code run faster by removing loops https://www.youtube.com/watch?v=lPPnGYjCSHY


def main():

    pygame.init()

    #Adjustable parameters
    number_of_cities = 5000
    population_size = 300
    mutation_rate = 0.03
    Generation_size = 60
    tournement_size = 5
    elite_size = 5

    #Creating dataframe from the csv provided
    df = pd.read_csv(r"C:\Users\smith\Downloads\traveling-santa-2018-prime-paths\cities.csv")  
    Cities = df.iloc[:number_of_cities]
    #order = Cities[['X', 'Y']].values.tolist()
    order = np.array(Cities[['X', 'Y']])
   

    #running the GA functions
    population = creating_population(order, population_size)
    best_order = Generations(population, Generation_size, tournement_size, mutation_rate, elite_size)
    
    #used to check where most computation time was spent to cut it down with numpy vectorizations
    cProfile.runctx(
        'Generations(population, Generation_size, tournement_size, mutation_rate, elite_size)',
       globals(),
        locals()
    )

    # creating pygame 

    screen = pygame.display.set_mode((800, 550))
    draw_circles(screen, order)
    draw_lines(best_order, screen)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


#repeat for n generations and return best result
def Generations(population, generations, tournement_size, mutation_rate, elite_size):
    
    best_ever_order = []
    best_ever_distance = float("inf")
    recorded_distances = []
    for generation in range(0, generations):

        distance = float("inf")
        #check if the last 2 recorded distances are less than 100 apart. if they are then it will run break_optima instead of the standard GA
        if len(recorded_distances) > 2:

            if recorded_distances[-1] > recorded_distances[-2]:
                distance = recorded_distances[-1] - recorded_distances[-2] 

            else:
                distance = recorded_distances[-2] - recorded_distances[-1]
        
        if distance < 100:
             population = break_local_optima(population, 3, 3, 0.18) 
        else:
            population = default_ga(population, elite_size, tournement_size, mutation_rate)

        #gets the current best order of the generation and calculates distance for it to display.
        best_order = get_best_order(population)
        total_distance = calculate_distance(best_order)

        recorded_distances.append(total_distance)

        #check to save the best recorded distance and order of all generations
        if total_distance <= min(recorded_distances):
            best_ever_order = best_order

        print("Generation", generation + 1,":", total_distance)
    print("Best Ever Distance:", min(recorded_distances))
    return best_ever_order

def break_local_optima(population, elite_size, tournement_size, mutation_rate):
    #sort population lowst to highest and select the specified size of elites to preserve
    sorted_population = collections.OrderedDict(sorted(population.items()))   
    elites = dict(list(sorted_population.items())[elite_size:])               

    #put the elites in population to preserve them
    population.update(elites)

    #Selection: Tournement
    selected = tournement_selection(population, tournement_size)
    # Crossover: Davis order
    children = crossover(selected, population)
    #Mutation: Inverse 
    mutated = mutate(children, mutation_rate)

    population = mutated 

    return population

def default_ga(population, elite_size, tournement_size, mutation_rate):

    #sort population lowst to highest and select the specified size of elites to preserve
    sorted_population = collections.OrderedDict(sorted(population.items()))   
    elites = dict(list(sorted_population.items())[elite_size:])              

    #put the elites in population to preserve them
    population.update(elites)

    #Selection: Roulette
    #selected = roulette_selection(population)

    #Selection Tournement
    selected = tournement_selection(population, tournement_size)
    # Crossover: Davis order
    children = crossover(selected, population)
    #Mutation: Inverse 
    mutated = mutate(children, mutation_rate)

    population = mutated 

    return population

def mutate(population, mutation_rate):
    mutated = []
    for order in population:
        #if a number less than mutation rate is picked then mutate the chromosome
        if random.random() < mutation_rate:
            chromosome = np.array(order)
            inversion_start = random.randint(0, len(chromosome) - 2)
            inversion_end = random.randint(inversion_start + 1, len(chromosome) - 1)
            #while loop to make sure that the end slice is always bigger than the start
            while inversion_end <= inversion_start:
                inversion_end = random.randint(inversion_start + 1, len(chromosome) - 1)

            #get the slice point from the chromsome and reverse it and add it back to the chromosome
            chromosome[inversion_start:inversion_end + 1] = chromosome[inversion_start:inversion_end + 1][::-1]
        
            mutated.append(chromosome)
        else:
            mutated.append(order)
          #calculate new fitness of mutated genes.
    mutated = np.array(mutated)
    mutated_fitness = calculate_fitness(mutated)
    return mutated_fitness

  
def crossover(selected, population_dict): 
    #Davis order crossover

    children = []
    for order in range(0, len(selected)):

        #create 2 parent
        parent_one = population_dict[random.choice(selected)]
        parent_two = population_dict[random.choice(selected)]
      
        #set a random crossover start and finish point to slice
        crossover_start = np.random.randint(0, len(parent_one) - 2)
        crossover_end = np.random.randint(crossover_start + 1, len(parent_one)-1)

        #while loop to make sure that the end slice is always bigger than the start
        while crossover_end <= crossover_start:
            crossover_end = np.random.randint(crossover_start + 1, len(parent_one) -1)

        #set length of child filled with 0s
        child = np.zeros_like(parent_one)
        #add the slice of parent one to the same index of the child
        parent_slice = parent_one[crossover_start:crossover_end]
        child[crossover_start:crossover_end] = parent_slice
        
        #I did have a while loop to add missing genes to the child however it took a majority of the time so i switched to a vectorization instead
        #np.isin(child,0) finds the elements of child array that are equal to 0 in a boolean array 
        #~np.isin(parent_two, child) inverts the function , instead checks the elemnts that are not in child are replaces the elements of 0 with them 
        child[np.isin(child, 0)] = parent_two[~np.isin(parent_two, child)]
        children.append(child)
    return np.array(children)



def roulette_selection(population):

    #gets list of all fitness values
    key_for_fitnesses = [(key, fitness) for (key, fitness) in population.keys()]
    fitnesses = [fitness for key, fitness in key_for_fitnesses]
    #gets sum of the values
    sum_of_fitnesses = sum(fitnesses)
    
    #list to store the picked from the roulette selection
    picked = []
    for i in range(0, len(population)):
        #gets a random value from 0 to sum of fitneses
        roullette = random.uniform(0, sum_of_fitnesses)
        #loop till counter is bigger or equal than the random number chosen
        current_sum = 0
        for key, fitness in population:
            current_sum += fitness
            if current_sum >= roullette:
                picked.append((key, fitness))
                break
    return picked


def tournement_selection(population, size): 

    #bracket to hold the winners
    best_bracket = []
    #loop through the "populations" amount of times to get the correct amount of values back
    for order in range(0, len(population)):
        #list to store current tournement
        bracket = []
        #loop for the size amount of times to fill the bracket up with a fitness value from the dictionary
        for j in range(0, size):
            #gets a random fitness and then appends it to the current bracket   
            current = random.choice(list(population.keys()))
            bracket.append(current)  
            #once the bracket has reached the size stated then get the best value and add it to the best_bracket
            if len(bracket) == size:
                best = max(bracket, key=lambda x: x[1])
                best_bracket.append(best)
                #clears the bracket for the next iteration.
                bracket.clear()
    return best_bracket

def get_best_order(population):
  #returns the best order array from the max (best) fitness value .
    best_fitness = max(population.keys())
    best_order = population[best_fitness]
    return best_order
    
@jit(nopython=True)
def calculate_distance(order: np.ndarray) -> np.ndarray:
    total_distance = 0

    for city in range(len(order)):
        current_city = order[city]
        #when the last city in the list is reached the next city will be the beginning of index 
        next_city = order[(city + 1) % len(order)]
        #distance between current and next city 
        distance = euclidean_distance(current_city, next_city)
        
        total_distance += distance

    return total_distance
                

def calculate_fitness(population):
    #calculate each distance of an order in the population
    distances = np.array([calculate_distance(order) for order in population])
    #the best distances need to have a higher fitness value.
    fitness = 1 / distances
    #the keys for the dictionary will be made up of a tuple (unique key, fitness) so it can include duplicate values. 
    keys = [(i, value) for i, value in enumerate(fitness)]
    # add the corresponding key to population index to the dictionary.
    fitness_population = dict(zip(keys, population))

    return fitness_population
        

def creating_population(order, size): 
    #create "size" number of random orders of cities , this is the population 
    #permutation shuffles the order for the amount of times of "size"
    population = [np.random.permutation(order) for i in range(0, size)]
    #Calculate the fitness of the population returns a dictionary with the key of fitness and value of order.
    population_fitness = calculate_fitness(population)
    
    return population_fitness


#the euclidean distance function from class live coding session chnaged to numpy to run faster
@jit(nopython=True)
def euclidean_distance(pointx: np.ndarray, pointy: np.ndarray) -> float:
    return np.sqrt(np.sum((pointx - pointy) ** 2))


def draw_lines(best_order, screen):
    #scaled down as the cities are scaled to fit on screen so lines are aswell to match them.
    scaler = 7
    #draw lines from the best order
    for city in range(len(best_order)):
        current_city = best_order[city]
        if city == len(best_order)-1:
            next_city = best_order[0]
        else:
             next_city = best_order[city + 1]

        current_city = [current_city[0] / scaler, current_city[1] / scaler]
        next_city = [next_city[0] / scaler, next_city[1] / scaler]

        pygame.draw.line(screen, (0, 200, 0), current_city, next_city)
        pygame.display.update()
        pygame.time.wait(10)

def draw_circles(screen, order):
    # represent cities as circles on screen
    #scaled down as not all the cities could be displayed on screen
    scaler = 7
    for city in order:
        x = city[0] / scaler
        y = city[1] / scaler
        pygame.draw.circle(screen, (0, 0, 255), (x, y), 1)


if __name__ == "__main__":
    main()
    