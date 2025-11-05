import streamlit as st 
import csv
import random
import pandas as pd # Import pandas for data frame display

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            # Skip the header
            header = next(reader)
            
            for row in reader:
                # Skip the line if it's empty
                if not row:
                    continue
                
                program = row[0]
                # Ensure the remaining columns are present before slicing
                if len(row) > 1:
                    # Convert ratings to floats. Handle potential ValueError defensively.
                    try:
                        ratings = [float(x) for x in row[1:]] 
                        program_ratings[program] = ratings
                    except ValueError:
                        st.error(f"Error reading ratings for program '{program}'. Ensure all ratings are numbers.")
                        return {}
    except FileNotFoundError:
        st.error(f"Error: CSV file not found at path: {file_path}. Please ensure 'program_ratings.csv' is in the same directory.")
        return {}
        
    return program_ratings

# Path to the CSV file
file_path = 'program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

# Fixed parameters
GEN = 100
POP = 50
EL_S = 2

all_programs = list(ratings.keys()) # all programs
all_time_slots = list(range(6, 24)) # time slots

# Check for program and slot mismatch (crucial for permutation GA)
if len(all_programs) != len(all_time_slots):
    st.error(f"FATAL ERROR: The number of programs ({len(all_programs)}) must equal the number of time slots ({len(all_time_slots)}) for a valid permutation schedule.")
    st.stop()
    
######################################### DEFINING FUNCTIONS ########################################################################
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    # Iterate over the shorter length to avoid index error if schedules are incomplete
    schedule_len = min(len(schedule), len(all_time_slots))
    for time_slot in range(schedule_len):
        program = schedule[time_slot]
        # ratings[program] is a list of 18 ratings
        total_rating += ratings[program][time_slot] 
    return total_rating

# initializing the population (only needs to return one initial schedule if we use random.shuffle later)
# We can use random.shuffle to generate the initial population, simplifying the brute force part.
def initialize_population_ga(programs, population_size):
    population = []
    
    # Create the first individual as the sorted program list, then shuffle copies
    base_schedule = list(programs)
    
    for _ in range(population_size):
        schedule = base_schedule.copy()
        random.shuffle(schedule)
        population.append(schedule)
        
    return population

# In permutation-based GA, we generate the initial population randomly, 
# so the previous finding_best_schedule/initialize_pop (brute force) is usually skipped
# as it's computationally prohibitive for 18! permutations.

# Create one initial random schedule (the best one from the initial population will be found in the GA loop)
initial_schedule = initialize_population_ga(all_programs, 1)[0]


############################################# GENETIC ALGORITHM (PERMUTATION-SAFE) #############################################################################

# Crossover: Order Crossover (OX1) for permutation schedules
def crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [''] * size, [''] * size
    
    # Select two random cut points
    start, end = sorted(random.sample(range(size), 2))

    # 1. Copy the segment from parent1 to child1
    child1[start:end] = parent1[start:end]
    
    # 2. Fill the remaining genes in child1 using the order of parent2
    p2_index = 0
    for i in range(size):
        if child1[i] == '':
            # Find the next program in parent2 that hasn't been used in child1
            while parent2[p2_index] in child1:
                p2_index += 1
            child1[i] = parent2[p2_index]
            p2_index += 1

    # Repeat for child2 (copy segment from parent2, fill from parent1)
    child2[start:end] = parent2[start:end]
    
    p1_index = 0
    for i in range(size):
        if child2[i] == '':
            while parent1[p1_index] in child2:
                p1_index += 1
            child2[i] = parent1[p1_index]
            p1_index += 1
            
    return child1, child2

# Mutation: Swap Mutation for permutation schedules
def mutate(schedule):
    # Select two random points and swap the programs
    mutation_points = random.sample(range(len(schedule)), 2)
    p1, p2 = mutation_points[0], mutation_points[1]
    schedule[p1], schedule[p2] = schedule[p2], schedule[p1]
    return schedule


def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=0.8, mutation_rate=0.02, elitism_size=EL_S):

    # Initialize the population randomly
    population = initialize_population_ga(initial_schedule, population_size)

    for generation in range(generations):
        # Calculate fitness for selection/sorting
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population = []

        # Elitism: Carry over the best individuals
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            # Selection: Tournament or simple random choice (using random.choices)
            parent1, parent2 = random.choices(population, k=2)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        # Trim population back to size if needed (due to extending with two children)
        population = new_population[:population_size] 

    # Return the best schedule found after all generations
    population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
    return population[0]

##################################################### TRIAL EXECUTION FUNCTION #####################################################################

def run_ga_trial(trial_name, co_r, mut_r):
    """Executes the Genetic Algorithm and returns the results for display."""
    
    # Run the GA for optimization using the initial random schedule as the starting point
    final_schedule = genetic_algorithm(
        initial_schedule, 
        generations=GEN, 
        population_size=POP, 
        crossover_rate=co_r, 
        mutation_rate=mut_r, 
        elitism_size=EL_S
    )
    
    # 3. Calculate Fitness
    total_ratings = fitness_function(final_schedule)
    
    # 4. Format Results
    schedule_data = []
    for time_slot, program in enumerate(final_schedule):
        time_str = f"{all_time_slots[time_slot]:02d}:00"
        rating = ratings[program][time_slot]
        schedule_data.append({
            "Time Slot": time_str,
            "Program": program,
            "Expected Rating": f"{rating:.3f}"
        })

    return final_schedule, total_ratings, schedule_data

##################################################### STREAMLIT INTERFACE #####################################################################

st.set_page_config(layout="wide")
st.title("📺 GA Scheduling Optimization - Multi-Trial Analysis")

# --- DISPLAY INITIAL DATA TABLE ---
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

# --- Parameter Inputs in Sidebar ---
for i in range(1, 4):
    trial_name = f"Trial {i}"
    
    st.sidebar.subheader(f"___{trial_name} Settings___")
    
    # Crossover Rate (CO_R) - Default values slightly varied
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

    # Mutation Rate (MUT_R) - Default values slightly varied
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

##################################################### MAIN EXECUTION & DISPLAY #####################################################################

st.header("Results and Analysis")

# --- Main Body Display Loop ---
for i, (co_r, mut_r) in enumerate(trial_params, 1): # Iterate over the collected parameters
    
    trial_name = f"Trial {i}"
    
    # Use the expander structure as requested
    with st.expander(f"**{trial_name} Results (CO_R: {co_r:.2f}, MUT_R: {mut_r:.2f})**", expanded=(i == 1)):
        
        # 1. Execute the GA Trial
        final_schedule, total_ratings, schedule_data = run_ga_trial(trial_name, co_r, mut_r)
        
        # 2. Display Results
        st.subheader(f"Optimal Schedule")
        
        # Display the Total Rating (Required output)
        st.metric("Total Expected Audience Ratings", f"{total_ratings:.3f}")
        
        # Display the Schedule Table (Required output)
        st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)

st.subheader("Implementation Details")
st.caption("Permutation-Safe Genetic Algorithm Operators:")
st.code("""
# Crossover: Order Crossover (OX1) is used to ensure no program is duplicated or omitted.
# Mutation: Swap Mutation is used to ensure the schedule remains a valid permutation.
# Initialization: Brute force initialization is replaced with random permutation initialization
#                 as brute force is infeasible for 18! schedules.
""")
