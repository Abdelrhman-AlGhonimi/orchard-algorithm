import xml.etree.ElementTree as ET
import collections
import random
import sys
import numpy as np
from datetime import datetime

# قراءة البيانات من ملف XML
def read_xml_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    operations = []
    surgeons = set()
    rooms = set()
    
    # قراءة وصف الملف
    description = root.find('description').text
    lines = description.split('\n')  # لا نستخدم strip() لتجنب فقدان المسافات
    
    # استخراج بيانات العمليات المؤجلة
    in_deferred_section = False
    for line in lines:
        if line.strip().startswith('Defered patients:'):
            in_deferred_section = True
            continue
        if line.strip() and in_deferred_section:
            try:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0].startswith('Surgeon'):
                    surgeon = int(parts[1])
                    patient = int(parts[3])
                    duration = int(parts[5])
                    emergency = parts[7] == 'True'
                    operations.append({
                        'surgeon': surgeon,
                        'patient': patient,
                        'duration': duration,
                        'emergency': emergency
                    })
                    surgeons.add(surgeon)
            except (ValueError, IndexError):
                continue
        
        # استخراج بيانات توزيع المرضى على الغرف
        if line.strip().startswith('Patient in Room'):
            try:
                # نستخرج الرقم بعد "Room"
                room_num = line.split('Room')[1].split(':')[0].strip()
                if room_num.isdigit():
                    room = int(room_num)
                    rooms.add(room)
            except (ValueError, IndexError):
                continue
    
    # استخراج بيانات جدولة العمليات
    for line in lines:
        if line.strip().startswith('Day') and 'Room' in line:
            try:
                parts = line.strip().split()
                day = int(parts[1])
                room = int(parts[3])
                # يمكن استخدام هذه المعلومات لتحديد أوقات البدء الممكنة
            except (ValueError, IndexError):
                continue
    
    # ترتيب الغرف
    rooms = sorted(list(rooms))
    
    return {
        'operations': operations,
        'surgeons': list(surgeons),
        'rooms': rooms,
        'num_rooms': len(rooms)
    }

# Objective function
def objective_function(solution, data):
    """Calculate the quality of the schedule"""
    operations = data['operations']
    rooms = data['rooms']
    day_duration = 480  # 8 hours work = 480 minutes
    
    # Initialize penalties
    emergency_wait = 0
    idle_time = 0
    overlap = 0
    room_usage = 0
    surgeon_conflict = 0
    
    # Create schedule for each room
    room_schedule = {room: [] for room in rooms}
    surgeon_schedule = {surgeon: [] for surgeon in data['surgeons']}
    
    # Sort operations by start time
    for i, (room, start_time) in enumerate(solution):
        op = operations[i]
        duration = op['duration']
        end_time = start_time + duration
        
        # Check for time overlaps in room
        for existing_op in room_schedule[room]:
            if (start_time < existing_op[1] and end_time > existing_op[0]):
                overlap += min(end_time, existing_op[1]) - max(start_time, existing_op[0])
        
        # Check for surgeon conflicts
        for existing_op in surgeon_schedule[op['surgeon']]:
            if (start_time < existing_op[1] and end_time > existing_op[0]):
                surgeon_conflict += min(end_time, existing_op[1]) - max(start_time, existing_op[0])
        
        # Add operation to schedules
        room_schedule[room].append((start_time, end_time, op['emergency']))
        surgeon_schedule[op['surgeon']].append((start_time, end_time))
        
        # Track room usage
        room_usage += duration
    
    # Calculate penalties
    for room in rooms:
        schedule = sorted(room_schedule[room])
        
        # Calculate emergency wait time
        for i, (start_time, end_time, emergency) in enumerate(schedule):
            if emergency:
                if i > 0:
                    prev_end = schedule[i-1][1]
                    emergency_wait += max(0, start_time - prev_end)
        
        # Calculate idle time in room
        if len(schedule) > 1:
            for i in range(len(schedule) - 1):
                end1 = schedule[i][1]
                start2 = schedule[i+1][0]
                idle_time += max(0, start2 - end1)
    
    # Calculate penalty weights
    emergency_weight = 4.0  # Higher priority for emergency cases
    idle_weight = 2.0       # Reduced weight for idle time
    overlap_weight = 5.0    # Very high penalty for overlaps
    usage_weight = 1.0      # Penalty for high room usage
    surgeon_weight = 3.0    # Penalty for surgeon conflicts
    
    # Calculate total score
    score = (-emergency_weight * emergency_wait - 
             idle_weight * idle_time - 
             overlap_weight * overlap -
             usage_weight * (room_usage - (len(operations) * 30)) -  # Target 30 minutes per operation
             surgeon_weight * surgeon_conflict)
    
    return score

# Initialize a random solution
def initialize_seedling(data):
    """Create an initial solution for scheduling operations with priority improvement"""
    solution = []
    operations = data['operations']
    rooms = data['rooms']
    day_duration = 480  # 8 hours work = 480 minutes
    
    # Sort operations by priority
    # 1. Emergency cases
    # 2. Longest duration operations
    operations_sorted = sorted(operations, 
                             key=lambda x: (-x['emergency'], -x['duration']))
    
    # Initialize start times for each room
    room_start_times = {room: 0 for room in rooms}
    room_usage = {room: 0 for room in rooms}  # To track room usage
    
    for op in operations_sorted:
        # Select the least used room
        available_rooms = sorted(rooms, key=lambda r: room_usage[r])
        room = available_rooms[0]
        
        # Calculate start time
        start_time = room_start_times[room]
        duration = op['duration']
        end_time = start_time + duration
        
        # Check day limits
        if end_time > day_duration:
            start_time = 0
            end_time = duration
        
        # Update room times
        room_start_times[room] = end_time
        room_usage[room] += duration
        
        # Add operation to solution
        solution.append((room, start_time))
    
    return solution

# Local search for a better schedule
def local_search(solution, data):
    """Local search for a better schedule"""
    rooms = data['rooms']
    operations = data['operations']
    
    # Copy the current solution
    neighbor = solution[:]
    
    # Select a random operation
    op_idx = random.randint(0, len(solution) - 1)
    current_op = operations[op_idx]
    
    # Get current room and time
    current_room, current_time = neighbor[op_idx]
    
    # 30% chance to change the room
    if random.random() < 0.3:
        # Choose a different room
        new_room = random.choice([r for r in rooms if r != current_room])
        neighbor[op_idx] = (new_room, current_time)
    
    # 30% chance to change the start time
    elif random.random() < 0.6:
        # Change the time by a random amount between -2 and +2 hours
        time_change = random.randint(-2, 2)
        new_time = max(0, min(24, current_time + time_change))
        neighbor[op_idx] = (current_room, new_time)
    
    # 20% chance to swap with another operation
    elif random.random() < 0.8:
        other_idx = random.randint(0, len(solution) - 1)
        if other_idx != op_idx:
            neighbor[op_idx], neighbor[other_idx] = neighbor[other_idx], neighbor[op_idx]
    
    # 20% chance to perform a block move
    else:
        # Select a random block size (2-5 operations)
        block_size = random.randint(2, 5)
        start_idx = random.randint(0, len(solution) - block_size)
        end_idx = start_idx + block_size
        
        # Move the block to a new position
        block = neighbor[start_idx:end_idx]
        new_pos = random.randint(0, len(solution) - block_size)
        
        # Insert block at new position
        neighbor = neighbor[:start_idx] + neighbor[end_idx:]
        neighbor[new_pos:new_pos] = block
    
    return neighbor

# Graft two schedules (cross-pollination)
def graft(strong_seedling, medium_seedling, data):
    """Graft two schedules (cross-pollination)"""
    new_solution = strong_seedling[:]
    operations = data['operations']
    
    # Calculate number of operations to graft (20-40% of total)
    num_to_graft = random.randint(int(len(new_solution) * 0.2), int(len(new_solution) * 0.4))
    
    # Select random operations to graft
    indices = random.sample(range(len(new_solution)), num_to_graft)
    
    # Replace selected operations with a mix of both solutions
    for idx in indices:
        if random.random() < 0.7:  # 70% chance to take from medium seedling
            new_solution[idx] = medium_seedling[idx]
        else:  # 30% chance to keep from strong seedling
            continue
    
    # Try to improve the solution by swapping operations
    for _ in range(3):  # Try 3 swaps
        if random.random() < 0.5:  # 50% chance to perform a swap
            idx1 = random.choice(indices)
            idx2 = random.choice(indices)
            if idx1 != idx2:
                new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
    
    return new_solution

# Orchard algorithm implementation (modified screening)
def orchard_algorithm(data, N=50, GYN=10, alpha=0.6, beta=0.4, 
                      num_strong_screening=10, num_weak_screening=10, num_grafting=5,
                      max_iterations=100, verbose=True):
    # Check data validity
    if verbose:
        print("\nStarting the algorithm...")
        print(f"Number of operations: {len(data['operations'])}")
        print(f"Number of rooms: {data['num_rooms']}")
        print(f"Number of surgeons: {len(data['surgeons'])}")
    
    # Check for operations
    if len(data['operations']) == 0:
        print("Error: No operations!")
        return None, None
    
    # Check for rooms
    if data['num_rooms'] == 0:
        print("Error: No rooms available!")
        return None, None
    
    # Initialize solutions
    solutions = []
    for _ in range(N):
        solution = initialize_seedling(data)
        fitness = objective_function(solution, data)
        solutions.append((solution, fitness))
    
    # Sort solutions by quality
    solutions.sort(key=lambda x: x[1], reverse=True)
    
    # Current best solution
    best_solution = solutions[0][0]
    best_fitness = solutions[0][1]
    
    # Create lists for strong and medium solutions
    strong_seedlings = solutions[:num_strong_screening]
    medium_seedlings = solutions[num_strong_screening:num_strong_screening+num_weak_screening]
    
    # Start iterations
    for iteration in range(max_iterations):
        if verbose:
            print(f"\nIteration {iteration+1}/{max_iterations}")
            print(f"Best fitness: {best_fitness}")
        
        # Grafting (cross-pollination)
        for _ in range(num_grafting):
            strong_idx = random.randint(0, len(strong_seedlings)-1)
            medium_idx = random.randint(0, len(medium_seedlings)-1)
            
            strong_seedling = strong_seedlings[strong_idx][0]
            medium_seedling = medium_seedlings[medium_idx][0]
            
            new_solution = graft(strong_seedling, medium_seedling, data)
            new_fitness = objective_function(new_solution, data)
            
            # Add new solution
            solutions.append((new_solution, new_fitness))
            
            # Update best solution
            if new_fitness > best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                if verbose:
                    print(f"Updated best solution: {best_fitness}")
        
        # Sort solutions
        solutions.sort(key=lambda x: x[1], reverse=True)
        solutions = solutions[:N]
        
        # Update lists
        strong_seedlings = solutions[:num_strong_screening]
        medium_seedlings = solutions[num_strong_screening:num_strong_screening+num_weak_screening]
    
    return best_solution, best_fitness

if __name__ == "__main__":
    # Read data from XML file
    try:
        data = read_xml_data('1_3PLInfo.xml')
        print(f"\nData read successfully. Number of operations: {len(data['operations'])}")
        print("\nOperations data:")
        for op in data['operations']:
            print(f"Surgeon {op['surgeon']}: Patient {op['patient']}, Duration {op['duration']}, Emergency: {op['emergency']}")
        print(f"\nNumber of rooms: {data['num_rooms']}")
        print(f"Available rooms: {data['rooms']}")
        print(f"Number of surgeons: {len(data['surgeons'])}")
    except Exception as e:
        print(f"Error reading XML file: {e}")
        exit(1)

    # Run the algorithm
    start_time = datetime.now()
    best_solution, best_fitness = orchard_algorithm(data)
    end_time = datetime.now()

    if best_solution is not None:
        print(f"\nFinal result:")
        print(f"Best schedule: {best_fitness}")
        print(f"Execution time: {end_time - start_time}")
        
        # Print final schedule