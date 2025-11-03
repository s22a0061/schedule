import streamlit as st # Add this line at the very top
import csv
import random
# ... (rest of your original code) ...

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
