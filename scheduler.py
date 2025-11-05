import streamlit as st # Add this line at the very top
import csv
import random
import pandas as pd

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

##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

# --- REMOVED REDUNDANT GLOBAL CO_R AND MUT_R SLIDERS HERE ---

# Other fixed parameters (KEEP THESE)
GEN = 100
POP = 50
EL_S = 2

all_programs = list(ratings.keys()) # all programs
all_time_slots = list(range(6, 24)) # time slots

######################################### STREAMLIT INTERFACE SETUP ########################################################################

st.title("📺 GA Scheduling Optimization - Multi-Trial Analysis")

# --- NEW: DISPLAY INITIAL DATA TABLE ---
st.header("📊 Original Program Rating Data")
st.caption("Ratings (audience share) for each program across all 18 time slots (Hour 6 - Hour 23).")

# Prepare the data for display (Transpose the dictionary for better readability)
df_ratings = pd.DataFrame(ratings).transpose()
df_ratings.columns = [f"{h:02d}:00" for h in all_time_slots]
st.dataframe(df_ratings, height=250, use_container_width=True)

# ----------------------------------------

st.sidebar.title("⚙️ GA Trial Parameters")
st.sidebar.caption("Adjust the Crossover and Mutation rates for each trial.")

# List to hold the selected parameters for each trial
trial_params = []

# --- Move Parameter Inputs to Sidebar (THIS IS CORRECTLY IMPLEMENTED) ---
for i in range(1, 4):
    trial_name = f"Trial {i}"
    
    st.sidebar.subheader(f"___{trial_name} Settings___")
    
    # Set different default values for demonstrative effect
    default_co_r = [0.8, 0.6, 0.9][i-1]
    co_r = st.sidebar.slider(
        'Crossover Rate (CO_R)', 
        min_value=0.0, 
        max_value=0.95, 
        value=default_co_r, 
        step=0.05,
        key=f'co_r_{i}',
        format='%.2f'
    )

    default_mut_r = [0.02, 0.05, 0.01][i-1]
    mut_r = st.sidebar.slider(
        'Mutation Rate (MUT_R)', 
        min_value=0.01, 
        max_value=0.05, 
        value=default_mut_r, 
        step=0.01,
        key=f'mut_r_{i}',
        format='%.2f'
    )
    
    trial_params.append((co_r, mut_r))
    
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
    # NOTE: This brute-force population initialization is extremely slow if len(programs) > 10.
    # It generates len(programs)! permutations.
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
# NOTE: The execution may stall here if you have more than ~10 programs.
all_possible_schedules = initialize_pop(all_programs, all_time_slots)

# callin the schedule func.
best_schedule = finding_best_schedule(all_possible_schedules)


############################################# GENETIC ALGORITHM #############################################################################

# Crossover - *** LOGICAL ERROR: Does not guarantee a valid permutation (can introduce duplicates/misses) ***
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutating - *** LOGICAL ERROR: Does not guarantee a valid permutation (can introduce duplicates) ***
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# calling the fitness func.
def evaluate_fitness(schedule):
    return fitness_function(schedule)

# genetic algorithms with parameters

def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate, mutation_rate, elitism_size=EL_S):

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

##################################################### TRIAL EXECUTION FUNCTION #####################################################################
# ... (run_ga_trial function remains unchanged) ...

##################################################### MAIN EXECUTION & DISPLAY #####################################################################

st.header("Results and Analysis")

# --- Main Body Display Loop ---
for i, (co_r, mut_r) in enumerate(trial_params, 1): # Iterate over the collected parameters
    
    trial_name = f"Trial {i}"
    
    # Use the expander structure as requested
    with st.expander(f"**{trial_name} Results (CO_R: {co_r:.2f}, MUT_R: {mut_r:.2f})**", expanded=(i == 1)):
        
        # 1. Execute the GA Trial
        # NOTE: It is assumed that all_possible_schedules and initial_best_schedule are computed once globally before this loop.
        final_schedule, total_ratings, schedule_data = run_ga_trial(trial_name, co_r, mut_r)
        
        # 2. Display Results
        # Parameter documentation is integrated into the expander title for clarity.
        st.subheader(f"Optimal Schedule")
        
        # Display the Total Rating (Required output)
        st.metric("Total Expected Audience Ratings", f"{total_ratings:.3f}")
        
        # Display the Schedule Table (Required output)
        st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)

st.subheader("Implementation Details")
st.code("""
# Fitness Function: Maximizes the sum of expected ratings for the schedule.
# Selection: Simple Elitism + random choice for crossover.
# Crossover: Single-point crossover.
# Mutation: Randomly replaces one program with another from the available programs.
""")
