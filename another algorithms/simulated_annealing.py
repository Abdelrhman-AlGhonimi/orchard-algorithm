import random
import math
import prim
from datetime import datetime

def simulated_annealing(data, initial_temperature=1000, cooling_rate=0.99, max_iterations=1000):
    """Simulated Annealing algorithm for surgical scheduling"""
    operations = data['operations']
    rooms = data['rooms']
    
    # Generate initial random solution
    current_solution = []
    for op in operations:
        room = random.choice(rooms)
        start_time = random.randint(480, 960 - op['duration'])
        current_solution.append((room, start_time))
    
    # Calculate initial fitness
    current_fitness = prim.objective_function(current_solution, data)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    temperature = initial_temperature
    
    for iteration in range(max_iterations):
        # Generate neighbor solution
        neighbor = current_solution.copy()
        
        # Select random operation to modify
        idx = random.randint(0, len(operations) - 1)
        op = operations[idx]
        
        # Change room or start time
        if random.random() < 0.5:
            # Change room
            current_room = neighbor[idx][0]
            available_rooms = [r for r in rooms if r != current_room]
            new_room = random.choice(available_rooms)
            neighbor[idx] = (new_room, neighbor[idx][1])
        else:
            # Change start time
            current_time = neighbor[idx][1]
            time_change = random.randint(-30, 30)  # Change by up to 30 minutes
            new_time = max(480, min(960 - op['duration'], current_time + time_change))
            neighbor[idx] = (neighbor[idx][0], new_time)
        
        # Calculate neighbor fitness
        neighbor_fitness = prim.objective_function(neighbor, data)
        
        # Calculate acceptance probability
        delta_fitness = neighbor_fitness - current_fitness
        if delta_fitness > 0 or random.random() < math.exp(delta_fitness / temperature):
            current_solution = neighbor
            current_fitness = neighbor_fitness
            
            # Update best solution if improved
            if neighbor_fitness > best_fitness:
                best_solution = neighbor
                best_fitness = neighbor_fitness
        
        # Cool down
        temperature *= cooling_rate
        
    return best_solution, best_fitness

def main():
    try:
        # Read data from XML file
        data = prim.read_xml_data('1_3PLInfo.xml')
        print(f"\nData read successfully. Number of operations: {len(data['operations'])}")
        print("\nOperations data:")
        for op in data['operations']:
            print(f"Surgeon {op['surgeon']}: Patient {op['patient']}, Duration {op['duration']}, Emergency: {op['emergency']}")
        print(f"\nNumber of rooms: {data['num_rooms']}")
        print(f"Available rooms: {data['rooms']}")
        print(f"Number of surgeons: {len(data['surgeons'])}")
        
        # Run simulated annealing
        start_time = datetime.now()
        best_solution, best_fitness = simulated_annealing(data)
        end_time = datetime.now()
        
        # Save results
        output = []
        output.append("Simulated Annealing Results:\n")
        output.append(f"Fitness: {best_fitness}\n")
        output.append(f"Execution time: {end_time - start_time}\n")
        
        # Print schedule
        output.append("\nSurgical Schedule:\n")
        output.append("==========================================================================\n")
        output.append("| Room | Surgeon | Patient | Duration | Emergency | Start        | End          |\n")
        output.append("==========================================================================\n")
        
        operations = data['operations']
        rooms = data['rooms']
        room_schedule = {room: [] for room in rooms}
        
        # Build schedule
        for i, (room, start_time) in enumerate(best_solution):
            op = operations[i]
            duration = op['duration']
            end_time = start_time + duration
            
            # Convert minutes to time format
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
        
        # Sort and print schedule
        for room in rooms:
            output.append(f"\nRoom {room}:")
            output.append("-" * 90 + "\n")
            for op in sorted(room_schedule[room], key=lambda x: x['start_time']):
                output.append(f"| {room:2d} | {op['surgeon']:7d} | {op['patient']:7d} | {op['duration']:8d} | {op['emergency']:9s} | {op['start_time']:12s} | {op['end_time']:12s} |\n")
            output.append("-" * 90 + "\n")
        
        # Save to file
        prim.save_to_file('simulated_annealing_schedule.txt', ''.join(output))
        
        print("\nResults saved to simulated_annealing_schedule.txt")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
