import csv
import random
import requests # Added for file download

# --- Data Loading Section (Inserted) ---
# URL of the raw CSV file
csv_url = 'https://raw.githubusercontent.com/s22a0061/schedule/refs/heads/main/program_ratings.csv'
file_path = 'program_ratings.csv'

try:
    # Download the file
    response = requests.get(csv_url)
    response.raise_for_status() # Check for bad status codes

    # Save the content to the expected file path
    with open(file_path, 'wb') as f:
        f.write(response.content)

    print(f"File downloaded successfully to {file_path}")
except requests.exceptions.RequestException as e:
    print(f"Error downloading file: {e}")
    # Handle the error or exit
# --- End of Data Loading Section ---

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)
        
        for row in reader:
            program = row[0]
            # Ensure there are enough ratings columns before converting
            try:
                ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
                program_ratings[program] = ratings
            except ValueError:
                 print(f"Skipping row for program {program} due to non-float rating value.")

    return program_ratings

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

# Print the result (you can also return or process it further)
print("\n--- Program Ratings Data ---")
for program, ratings in program_ratings_dict.items():
    print(f"'{program}': {ratings},")
print("----------------------------\n")


##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

GEN = 100
POP = 50
CO_R = 0.8
MUT_R = 0.2
EL_S = 2

all_programs = list(ratings.keys()) # all programs
# Adjusting all_time_slots based on the number of programs/ratings
# The time slots are 6:00 to 23:00, which is 18 slots (24-6=18).
# The ratings data has 18 values per program, so the indexing is 0 to 17.
all_time_slots = list(range(6, 24)) # 18 time slots

# Safety check: ensure number of programs matches number of time slots expected by initialize_pop
if len(all_programs) > len(all_time_slots):
    print("Warning: More programs than time slots. The brute-force search (initialize_pop) might be inappropriate or incomplete.")
    # For this specific dataset, len(all_programs) = 18 and len(all_time_slots) = 18, so this is okay.
    # The structure of the brute-force search implies 1-to-1 mapping.

######################################### DEFINING FUNCTIONS ########################################################################
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    # The time slot index (time_slot) corresponds to the index of the rating in the program's ratings list.
    for time_slot_index, program in enumerate(schedule): 
        # Safety check: prevent IndexError if schedule length exceeds available ratings
        if time_slot_index < len(ratings[program]):
             total_rating += ratings[program][time_slot_index]
    return total_rating

# initializing the population (Generates all possible permutations of programs)
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    # This function is computationally expensive (n!)
    for i in range(len(programs)):
        # Recursive call to generate permutations
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# selection (brute force search for the best schedule)
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = -float('inf') # Initialize to negative infinity

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

# Note on the Brute-Force section:
# With 18 programs, 18! is an extremely large number (6.4 x 10^15).
# The `initialize_pop` function will crash the environment due to memory/time constraints.
# The user's original logic attempts to run both Brute-Force and GA, which is not feasible for 18! permutations.

# Since the user's code structure *depends* on the `initial_best_schedule` from the brute-force search 
# to seed the GA, I must acknowledge that the brute-force part (`initialize_pop` and `finding_best_schedule`) 
# will **fail** in a standard execution environment.

# Given that the **Genetic Algorithm (GA)** part is designed to find a near-optimal solution without brute force, 
# I will modify the GA initialization to skip the infeasible brute-force step and use a **randomly generated initial schedule** # (which the GA itself does to populate the rest of the population).

# MODIFICATION: Skipping the computationally impossible Brute-Force search.
# Instead, the GA will start with a single, randomly shuffled schedule.

# initial_best_schedule = finding_best_schedule(all_possible_schedules) # THIS WAS OMITTED
initial_schedule_seed = all_programs.copy()
random.shuffle(initial_schedule_seed) 

# The remaining time slots logic is also based on the brute-force failure, which is now unnecessary.
# We will just run the GA starting from a random seed to find the best 18-slot schedule.
rem_t_slots = 0 # No remaining slots, as the initial schedule covers all 18.


############################################# GENETIC ALGORITHM #############################################################################

# Crossover (Uses uniform crossover by index, but may create invalid permutations)
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1_raw = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2_raw = schedule2[:crossover_point] + schedule1[crossover_point:]
    
    # Simple fix for invalid chromosomes (duplicate programs in a schedule)
    # This heuristic attempts to fill in missing programs from the "swapped" part.
    
    # Fix Child 1
    # Find duplicates in child1_raw
    seen = set()
    child1 = []
    for program in child1_raw:
        if program not in seen:
            seen.add(program)
            child1.append(program)
    
    # Find missing programs and add them to the end (maintaining length)
    missing_programs = [p for p in all_programs if p not in seen]
    random.shuffle(missing_programs)
    
    # Truncate child1 if it's too long, or fill with missing if too short
    if len(child1) < len(all_programs):
        child1.extend(missing_programs[:len(all_programs) - len(child1)])
    elif len(child1) > len(all_programs):
        child1 = child1[:len(all_programs)]

    # Fix Child 2 (similar logic)
    seen = set()
    child2 = []
    for program in child2_raw:
        if program not in seen:
            seen.add(program)
            child2.append(program)

    missing_programs = [p for p in all_programs if p not in seen]
    random.shuffle(missing_programs)
    
    if len(child2) < len(all_programs):
        child2.extend(missing_programs[:len(all_programs) - len(child2)])
    elif len(child2) > len(all_programs):
        child2 = child2[:len(all_programs)]

    # A better GA for permutations would be PMX, but for simplicity, using this repair heuristic.
    return child1, child2

# mutating (Performs a simple swap, preserving chromosome validity)
def mutate(schedule):
    # Swap two programs to maintain the set of programs (a valid permutation)
    idx1, idx2 = random.sample(range(len(schedule)), 2)
    schedule[idx1], schedule[idx2] = schedule[idx2], schedule[idx1]
    return schedule

# calling the fitness func.
def evaluate_fitness(schedule):
    return fitness_function(schedule)

# genetic algorithms with parameters
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):

    # 1. Initialize Population based on the initial_schedule_seed
    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule) # Create diversity
        population.append(random_schedule)

    best_overall_schedule = initial_schedule
    best_overall_fitness = fitness_function(initial_schedule)

    for generation in range(generations):
        new_population = []

        # 2. Evaluation & Selection (Elitism)
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        current_best_schedule = population[0]
        current_best_fitness = fitness_function(current_best_schedule)

        # Update overall best
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_schedule = current_best_schedule.copy()

        # Elitism
        new_population.extend([s.copy() for s in population[:elitism_size]]) 
        
        # 3. Crossover and Mutation
        while len(new_population) < population_size:
            # Tournament selection (selecting parents based on fitness)
            parent1, parent2 = random.choices(population, k=2)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation uses the corrected swap operation
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        # Truncate population to maintain size
        population = new_population[:population_size]

    return best_overall_schedule

##################################################### RESULTS ###################################################################################

# genetic algorithm with parameters
# initial_best_schedule is replaced by the initial_schedule_seed
genetic_schedule = genetic_algorithm(
    initial_schedule_seed, 
    generations=GEN, 
    population_size=POP, 
    elitism_size=EL_S
)

# The brute force logic was removed, so the final schedule is simply the result of the GA.
final_schedule = genetic_schedule

print("\n=======================================================")
print("Final Optimal Schedule (Found via Genetic Algorithm):")
print("=======================================================")
for time_slot_index, program in enumerate(final_schedule):
    # Map the index back to the time slot hour
    print(f"Time Slot {all_time_slots[time_slot_index]:02d}:00 - Program '{program}'")

print("-------------------------------------------------------")
print(f"Total Ratings: {fitness_function(final_schedule):.2f}")
print("=======================================================")
