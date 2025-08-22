import prim
import random
from datetime import datetime

def genetic_scheduling(data, generations=50, population_size=10):
    operations = data['operations']
    rooms = data['rooms']

    def create_individual():
        return [(random.choice(rooms), random.randint(480, 960 - op['duration'])) for op in operations]

    def fitness(individual):
        return prim.objective_function(individual, data)

    def crossover(parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]

    def mutate(individual):
        idx = random.randint(0, len(individual) - 1)
        room = random.choice(rooms)
        start = random.randint(480, 960 - operations[idx]['duration'])
        individual[idx] = (room, start)

    population = [create_individual() for _ in range(population_size)]

    for _ in range(generations):
        population.sort(key=fitness)
        next_gen = population[:2]  # Elitism
        while len(next_gen) < population_size:
            parents = random.sample(population[:5], 2)
            child = crossover(parents[0], parents[1])
            if random.random() < 0.2:
                mutate(child)
            next_gen.append(child)
        population = next_gen

    return population[0]

def main():
    try:
        data = prim.read_xml_data('1_3PLInfo.xml')
        print(f"\nData read successfully. Number of operations: {len(data['operations'])}")
        print(f"Number of rooms: {data['num_rooms']}")
        print(f"Available rooms: {data['rooms']}")
        print(f"Number of surgeons: {len(data['surgeons'])}")

        start_time = datetime.now()
        solution = genetic_scheduling(data)
        end_time = datetime.now()

        fitness = prim.objective_function(solution, data)

        output = []
        output.append("Genetic Algorithm Scheduling Results:\n")
        output.append(f"Fitness: {fitness}\n")
        output.append(f"Execution time: {end_time - start_time}\n")

        output.append("\nSurgical Schedule:\n")
        output.append("==========================================================================\n")
        output.append("| Room | Surgeon | Patient | Duration | Emergency | Start        | End          |\n")
        output.append("==========================================================================\n")

        operations = data['operations']
        rooms = data['rooms']
        room_schedule = {room: [] for room in rooms}

        for i, (room, start_time) in enumerate(solution):
            op = operations[i]
            duration = op['duration']
            end_time = start_time + duration

            start_hour = start_time // 60
            start_minute = start_time % 60
            end_hour = end_time // 60
            end_minute = end_time % 60

            start_time_str = f"{start_hour:02d}:{start_minute:02d}"
            end_time_str = f"{end_hour:02d}:{end_minute:02d}"

            room_schedule[room].append({
                'surgeon': op['surgeon'],
                'patient': op['patient'],
                'duration': duration,
                'emergency': 'Yes' if op['emergency'] else 'No',
                'start_time': start_time_str,
                'end_time': end_time_str
            })

        for room in rooms:
            output.append(f"\nRoom {room}:")
            output.append("-" * 90 + "\n")
            for op in sorted(room_schedule[room], key=lambda x: x['start_time']):
                output.append(f"| {room:2d} | {op['surgeon']:7d} | {op['patient']:7d} | {op['duration']:8d} | {op['emergency']:9s} | {op['start_time']:12s} | {op['end_time']:12s} |\n")
            output.append("-" * 90 + "\n")

        prim.save_to_file('genetic_scheduling_schedule.txt', ''.join(output))
        print("\nResults saved to genetic_scheduling_schedule.txt")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
