import random
import numpy as np
import matplotlib.pyplot as plt


NUM_JOBS = 15
NUM_VMS = 5
POPULATION_SIZE = 70
GENERATIONS = 100
MUTATION_RATE = 0.2
MAX_VM_LOAD = 50    # Maximum load per VM


def generate_job_times(num_jobs):
    return [random.randint(5, 20) for _ in range(num_jobs)]

def generate_vm_speeds(num_vms):
    return [random.uniform(0.5, 1.5) for _ in range(num_vms)]

job_times = generate_job_times(NUM_JOBS)
vm_speeds = generate_vm_speeds(NUM_VMS)
job_priorities = [random.randint(1, 5) for _ in range(NUM_JOBS)]  # 1 (low) to 5 (high)


def calculate_vm_loads(chromosome):
    vm_loads = [0.0] * NUM_VMS
    for job_idx, vm_idx in enumerate(chromosome):
        execution_time = job_times[job_idx] / vm_speeds[vm_idx]
        vm_loads[vm_idx] += execution_time
    return vm_loads

def calculate_makespan(chromosome):
    return max(calculate_vm_loads(chromosome))

def calculate_weighted_priority_score(chromosome):
    score = 0.0
    for job_idx, vm_idx in enumerate(chromosome):
        exec_time = job_times[job_idx] / vm_speeds[vm_idx]
        score += job_priorities[job_idx] / exec_time
    return score

def fitness(chromosome):
    vm_loads = calculate_vm_loads(chromosome)
    makespan = max(vm_loads)
    load_variance = np.var(vm_loads)
    overload_penalty = sum((max(0, load - MAX_VM_LOAD))**2 for load in vm_loads)
    priority_score = calculate_weighted_priority_score(chromosome)

    return priority_score / (1 + makespan + load_variance + overload_penalty)


def initial_population():
    return [
        [random.randint(0, NUM_VMS - 1) for _ in range(NUM_JOBS)]
        for _ in range(POPULATION_SIZE)
    ]

def selection(population):
    weights = [fitness(chrom) for chrom in population]
    return random.choices(population, weights=weights, k=2)

def crossover(parent1, parent2):
    return [random.choice([g1, g2]) for g1, g2 in zip(parent1, parent2)]

def mutate(chromosome):
    for i in range(NUM_JOBS):
        if random.random() < MUTATION_RATE:
            chromosome[i] = random.randint(0, NUM_VMS - 1)
    return chromosome

def plot_progress(makespan_history):
    plt.plot(makespan_history, marker='o')
    plt.title("Best Makespan Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Makespan")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_vm_loads(chromosome):
    vm_loads = calculate_vm_loads(chromosome)
    for i, load in enumerate(vm_loads):
        print(f"  VM {i} final load: {load:.2f}")


def genetic_algorithm():
    population = initial_population()
    best_schedule = None
    best_makespan = float('inf')
    makespan_history = []

    for gen in range(GENERATIONS):
        new_population = []
        best_gen = max(population, key=fitness)
        best_gen_makespan = calculate_makespan(best_gen)
        if best_gen_makespan < best_makespan:
            best_makespan = best_gen_makespan
            best_schedule = best_gen
        new_population.append(best_gen)

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = selection(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        makespan_history.append(best_makespan)

        if gen % 10 == 0 or gen == GENERATIONS - 1:
            print(f"Generation {gen:03d}: Best Makespan = {best_makespan:.2f}")

    plot_progress(makespan_history)
    return best_schedule, best_makespan


if __name__ == "__main__":
    print("Generated Job Times:", job_times)
    print("Generated Job Priorities:", job_priorities)
    print("Generated VM Speeds:", [round(s, 2) for s in vm_speeds])

    print("\nRunning Genetic Algorithm...\n")
    best_schedule, best_makespan = genetic_algorithm()

    print("\nOptimal Job Schedule (Job -> VM):")
    for idx, vm in enumerate(best_schedule):
        print(f"  Job {idx} (Priority {job_priorities[idx]}) -> VM {vm}")

    print(f"\nMinimum Makespan Achieved: {round(best_makespan, 2)}")
    display_vm_loads(best_schedule)
