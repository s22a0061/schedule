import streamlit as st # Add this line at the very top
import csv
import random

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)
        
        for row in reader:
            # --- ADDED CHECK HERE ---
            if not row:
                continue # Skip the line if it's empty
            # ------------------------
            
            program = row[0]
            # Ensure the remaining columns are present before slicing
            if len(row) > 1:
                ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
                program_ratings[program] = ratings
    
    return program_ratings

# Path to the CSV file
file_path = 'program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

# Print the result (you can also return or process it further)
for program, ratings in program_ratings_dict.items():
    print(f"'{program}': {ratings},")

##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

GEN = 100
POP = 50
CO_R = 0.8
MUT_R = 0.2
EL_S = 2

all_programs = list(ratings.keys()) # all programs
all_time_slots = list(range(6, 24)) # time slots

######################################### DEFINING FUNCTIONS ########################################################################
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# initializing the population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# selection
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

# calling the pop func.
all_possible_schedules = initialize_pop(all_programs, all_time_slots)

# callin the schedule func.
best_schedule = finding_best_schedule(all_possible_schedules)


############################################# GENETIC ALGORITHM #############################################################################

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# mutating
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# calling the fitness func.
def evaluate_fitness(schedule):
    return fitness_function(schedule)

# genetic algorithms with parameters



def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):

    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitsm
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

##################################################### RESULTS ###################################################################################

# brute force
initial_best_schedule = finding_best_schedule(all_possible_schedules)

# NOTE: The Brute Force section is computationally expensive and
# 'initial_best_schedule' will only contain programs for the length
# of 'all_programs'. For an 18-hour schedule, 'all_programs' must have 18 items.
# If 'all_programs' < 18, this section will lead to an incomplete schedule.

# Assuming your brute force is finding the best schedule for the programs available
# and you want the GA to fill the remaining time slots, as per your original logic:

rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
genetic_schedule = genetic_algorithm(initial_best_schedule, generations=GEN, population_size=POP, elitism_size=EL_S)

# Adjust 'final_schedule' to handle the case where brute force found the full schedule
if rem_t_slots > 0:
    final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]
else:
    # If initial_best_schedule is full (len=18), use the GA to optimize it further
    final_schedule = genetic_algorithm(initial_best_schedule, generations=GEN, population_size=POP, elitism_size=EL_S)


# **Replace your original print statements with Streamlit components**

st.title("📺 Optimal TV Scheduling with Genetic Algorithm")

st.header("Genetic Algorithm Parameters")
st.write(f"**Generations:** {GEN}")
st.write(f"**Population Size:** {POP}")
st.write(f"**Crossover Rate:** {CO_R}")
st.write(f"**Mutation Rate:** {MUT_R}")

st.header("Final Optimal Schedule")

schedule_data = []
for time_slot, program in enumerate(final_schedule):
    time_str = f"{all_time_slots[time_slot]:02d}:00"
    rating = ratings[program][time_slot]
    schedule_data.append({
        "Time Slot": time_str,
        "Program": program,
        "Expected Rating": f"{rating:.2f}"
    })

import pandas as pd
st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)

total_ratings = fitness_function(final_schedule)
st.metric("Total Expected Audience Ratings", f"{total_ratings:.2f}")

st.subheader("Implementation Details")
st.code("""
# Fitness Function: Maximizes the sum of expected ratings for the schedule.
# Selection: Simple Elitism + random choice for crossover.
# Crossover: Single-point crossover.
# Mutation: Randomly replaces one program with another from the available programs.
""")
