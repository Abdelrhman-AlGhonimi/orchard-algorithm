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
    # 3. Operations that require the same surgeon
    operations_sorted = sorted(operations, 
                             key=lambda x: (-x['emergency'], -x['duration'], x['surgeon']))
    
    # Initialize start times for each room
    room_start_times = {room: 0 for room in rooms}
    room_usage = {room: 0 for room in rooms}  # To track room usage
    surgeon_rooms = {surgeon: [] for surgeon in data['surgeons']}  # Track rooms for each surgeon
    
    # Group operations by surgeon
    surgeon_operations = {}
    for op in operations_sorted:
        if op['surgeon'] not in surgeon_operations:
            surgeon_operations[op['surgeon']] = []
        surgeon_operations[op['surgeon']].append(op)
    
    # Process operations
    for surgeon, ops in surgeon_operations.items():
        # Try to keep operations for the same surgeon in the same room
        if surgeon_rooms[surgeon]:
            preferred_room = random.choice(surgeon_rooms[surgeon])
        else:
            preferred_room = random.choice(rooms)
        
        for op in ops:
            # Try preferred room first
            if room_usage[preferred_room] + op['duration'] <= day_duration:
                room = preferred_room
            else:
                # Find the least used room that can accommodate the operation
                available_rooms = sorted(
                    [(r, room_usage[r]) for r in rooms if room_usage[r] + op['duration'] <= day_duration],
                    key=lambda x: x[1]
                )
                if available_rooms:
                    room = available_rooms[0][0]
                else:
                    # If no room is available, use the least used room
                    room = min(rooms, key=lambda r: room_usage[r])
            
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
            
            # Track surgeon's rooms
            surgeon_rooms[surgeon].append(room)
            
            # Add operation to solution
            solution.append((room, start_time))
    
    return solution

# Local search for a better schedule
def local_search(solution, data, gradual_improvement=True):
    """Local search for a better schedule with gradual improvement option"""
    rooms = data['rooms']
    operations = data['operations']
    
    # Copy the current solution
    neighbor = solution[:]
    
    if gradual_improvement:
        # Try multiple small improvements
        for _ in range(3):  # Try 3 small improvements
            if random.random() < 0.4:  # 40% chance for room change
                # Select operation with highest overlap
                max_overlap = 0
                op_idx = -1
                for i, (room, start_time) in enumerate(solution):
                    op = operations[i]
                    duration = op['duration']
                    end_time = start_time + duration
                    
                    # Calculate overlap with other operations in the same room
                    overlap = 0
                    for j, (other_room, other_start) in enumerate(solution):
                        if i != j and room == other_room:
                            other_end = other_start + operations[j]['duration']
                            if (start_time < other_end and end_time > other_start):
                                overlap += min(end_time, other_end) - max(start_time, other_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        op_idx = i
                
                if op_idx >= 0:
                    # Try to move this operation to a different room
                    current_room = solution[op_idx][0]
                    available_rooms = [r for r in rooms if r != current_room]
                    
                    # Sort rooms by usage
                    room_usage = {room: 0 for room in rooms}
                    for r, _ in solution:
                        room_usage[r] += 1
                    
                    # Try less used rooms first
                    sorted_rooms = sorted(available_rooms, key=lambda r: room_usage[r])
                    
                    # Try to find a room that reduces overlap
                    for new_room in sorted_rooms:
                        new_solution = solution[:]
                        new_solution[op_idx] = (new_room, solution[op_idx][1])
                        new_overlap = 0
                        
                        # Calculate new overlap
                        start_time = new_solution[op_idx][1]
                        end_time = start_time + operations[op_idx]['duration']
                        
                        for j, (other_room, other_start) in enumerate(new_solution):
                            if op_idx != j and new_room == other_room:
                                other_end = other_start + operations[j]['duration']
                                if (start_time < other_end and end_time > other_start):
                                    new_overlap += min(end_time, other_end) - max(start_time, other_start)
                        
                        if new_overlap < max_overlap:
                            neighbor = new_solution
                            break
            
            elif random.random() < 0.7:  # 30% chance for time adjustment
                # Select operation with highest idle time
                max_idle = 0
                op_idx = -1
                for i, (room, start_time) in enumerate(solution):
                    op = operations[i]
                    duration = op['duration']
                    end_time = start_time + duration
                    
                    # Calculate idle time with next operation
                    idle = 0
                    next_idx = i + 1
                    while next_idx < len(solution) and solution[next_idx][0] == room:
                        next_start = solution[next_idx][1]
                        if next_start > end_time:
                            idle = next_start - end_time
                            break
                        next_idx += 1
                    
                    if idle > max_idle:
                        max_idle = idle
                        op_idx = i
                
                if op_idx >= 0:
                    # Try to reduce idle time
                    current_time = solution[op_idx][1]
                    if current_time > 0:  # Can move earlier
                        new_time = max(0, current_time - 1)
                        new_solution = solution[:]
                        new_solution[op_idx] = (new_solution[op_idx][0], new_time)
                        neighbor = new_solution
            
            else:  # 30% chance for block move
                # Select block with highest overlap
                max_overlap = 0
                block_idx = -1
                for i in range(len(solution) - 2):
                    overlap = 0
                    for j in range(i, i+3):
                        op = operations[j]
                        duration = op['duration']
                        end_time = solution[j][1] + duration
                        
                        for k in range(i, i+3):
                            if j != k:
                                other_end = solution[k][1] + operations[k]['duration']
                                if (solution[j][1] < other_end and end_time > solution[k][1]):
                                    overlap += min(end_time, other_end) - max(solution[j][1], solution[k][1])
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        block_idx = i
                
                if block_idx >= 0:
                    # Try to move block to reduce overlap
                    block = solution[block_idx:block_idx+3]
                    new_pos = random.randint(0, len(solution) - 3)
                    
                    # Insert block at new position
                    new_solution = solution[:block_idx] + solution[block_idx+3:]
                    new_solution[new_pos:new_pos] = block
                    neighbor = new_solution
    
    else:
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
def graft(strong_seedling, medium_seedling, data, gradual_improvement=True):
    """Graft two schedules (cross-pollination) with gradual improvement option"""
    new_solution = strong_seedling[:]
    operations = data['operations']
    
    if gradual_improvement:
        # Calculate number of operations to graft (10-30% of total)
        num_to_graft = random.randint(int(len(new_solution) * 0.1), int(len(new_solution) * 0.3))
        
        # Select operations based on their quality
        quality_scores = []
        for i, (room, start_time) in enumerate(strong_seedling):
            op = operations[i]
            duration = op['duration']
            end_time = start_time + duration
            
            # Calculate quality score for this operation
            score = 0
            
            # Check overlap
            for j, (other_room, other_start) in enumerate(strong_seedling):
                if i != j and room == other_room:
                    other_end = other_start + operations[j]['duration']
                    if (start_time < other_end and end_time > other_start):
                        score -= min(end_time, other_end) - max(start_time, other_start)
            
            # Check idle time
            next_idx = i + 1
            while next_idx < len(strong_seedling) and strong_seedling[next_idx][0] == room:
                next_start = strong_seedling[next_idx][1]
                if next_start > end_time:
                    score -= (next_start - end_time)
                    break
                next_idx += 1
            
            quality_scores.append((i, score))
        
        # Sort operations by quality (worst first)
        quality_scores.sort(key=lambda x: x[1])
        
        # Select worst operations to potentially replace
        indices = [x[0] for x in quality_scores[:num_to_graft]]
        
        # Try to improve each selected operation
        for idx in indices:
            # Get current operation
            current_op = operations[idx]
            current_room, current_time = strong_seedling[idx]
            
            # Try to find a better room from medium seedling
            best_score = float('-inf')
            best_solution = None
            
            for other_idx in range(len(medium_seedling)):
                if other_idx != idx:
                    new_solution = strong_seedling[:]
                    new_solution[idx] = medium_seedling[other_idx]
                    
                    # Calculate new score
                    score = objective_function([new_solution[idx]], data)
                    
                    if score > best_score:
                        best_score = score
                        best_solution = new_solution
            
            if best_solution:
                new_solution = best_solution
        
        # Try to improve the solution by swapping operations
        for _ in range(2):  # Try 2 swaps
            if random.random() < 0.6:  # 60% chance to perform a swap
                # Select operations with highest overlap
                max_overlap = 0
                idx1, idx2 = -1, -1
                
                for i in range(len(new_solution) - 1):
                    for j in range(i + 1, len(new_solution)):
                        if new_solution[i][0] == new_solution[j][0]:
                            op1 = operations[i]
                            op2 = operations[j]
                            
                            start1 = new_solution[i][1]
                            end1 = start1 + op1['duration']
                            start2 = new_solution[j][1]
                            end2 = start2 + op2['duration']
                            
                            if (start1 < end2 and end1 > start2):
                                overlap = min(end1, end2) - max(start1, start2)
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    idx1, idx2 = i, j
                
                if idx1 >= 0 and idx2 >= 0:
                    # Try swapping these operations
                    new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
                    
                    # Check if swap improved the solution
                    if objective_function(new_solution, data) > objective_function(strong_seedling, data):
                        continue
                    else:
                        # If not, revert the swap
                        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
    
    else:
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

def print_schedule(solution, data):
    """Print the schedule in a readable format"""
    operations = data['operations']
    rooms = data['rooms']
    
    # Create schedule for each room
    room_schedule = {room: [] for room in rooms}
    
    # Sort operations by start time
    for i, (room, start_time) in enumerate(solution):
        op = operations[i]
        duration = op['duration']
        end_time = start_time + duration
        
        # Format time
        start_time_str = f"{start_time:02.0f}:{(start_time % 1) * 60:02.0f}"
        end_time_str = f"{end_time:02.0f}:{(end_time % 1) * 60:02.0f}"
        
        # Add to room schedule
        room_schedule[room].append({
            'surgeon': op['surgeon'],
            'patient': op['patient'],
            'duration': duration,
            'emergency': 'Yes' if op['emergency'] else 'No',
            'start_time': start_time_str,
            'end_time': end_time_str
        })
    
    # Print schedule for each room
    print("\nSurgical Schedule:")
    print("""==================================================
| Room | Surgeon | Patient | Duration | Emergency | Start | End |
==================================================""")
    
    for room in rooms:
        print(f"\nRoom {room}:")
        print("-" * 70)
        for op in sorted(room_schedule[room], key=lambda x: x['start_time']):
            print(f"| {room:2d} | {op['surgeon']:2d} | {op['patient']:2d} | {op['duration']:2d} | {op['emergency']:3s} | {op['start_time']} | {op['end_time']} |")
        print("-" * 70)

def save_to_file(filename, content):
    """Save content to a text file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nResults have been saved to {filename}")
    except Exception as e:
        print(f"Error saving to file: {e}")

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
        # Create output content
        output = []
        output.append(f"Data read successfully. Number of operations: {len(data['operations'])}\n")
        output.append("\nOperations data:\n")
        for op in data['operations']:
            output.append(f"Surgeon {op['surgeon']}: Patient {op['patient']}, Duration {op['duration']}, Emergency: {op['emergency']}\n")
        output.append(f"\nNumber of rooms: {data['num_rooms']}\n")
        output.append(f"Available rooms: {data['rooms']}\n")
        output.append(f"Number of surgeons: {len(data['surgeons'])}\n")
        
        # Run the algorithm once
        best_solution, best_fitness = orchard_algorithm(data)
        
        # Add algorithm results
        output.append("\nAlgorithm results:\n")
        output.append(f"Final fitness: {best_fitness}\n")

        # Final results
        output.append("\nFinal result:\n")
        output.append(f"Best schedule: {best_fitness}\n")
        output.append(f"Execution time: {end_time - start_time}\n")
        
        # Schedule
        output.append("\nSurgical Schedule:\n")
        output.append("==================================================\n")
        output.append("| Room | Surgeon | Patient | Duration | Emergency | Start | End |\n")
        output.append("==================================================\n")
        
        operations = data['operations']
        rooms = data['rooms']
        room_schedule = {room: [] for room in rooms}
        
        # Sort operations by start time
        for i, (room, start_time) in enumerate(best_solution):
            op = operations[i]
            duration = op['duration']
            end_time = start_time + duration
            
            # Format time
            start_time_str = f"{start_time:02.0f}:{(start_time % 1) * 60:02.0f}"
            end_time_str = f"{end_time:02.0f}:{(end_time % 1) * 60:02.0f}"
            
            # Add to room schedule
            room_schedule[room].append({
                'surgeon': op['surgeon'],
                'patient': op['patient'],
                'duration': duration,
                'emergency': 'Yes' if op['emergency'] else 'No',
                'start_time': start_time_str,
                'end_time': end_time_str
            })
        
        # Print schedule for each room
        for room in rooms:
            output.append(f"\nRoom {room}:\n")
            output.append("-" * 70 + "\n")
            for op in sorted(room_schedule[room], key=lambda x: x['start_time']):
                output.append(f"| {room:2d} | {op['surgeon']:2d} | {op['patient']:2d} | {op['duration']:2d} | {op['emergency']:3s} | {op['start_time']} | {op['end_time']} |\n")
            output.append("-" * 70 + "\n")

        # Save to file
        save_to_file('surgical_schedule.txt', ''.join(output))
        
        # Print final results to console
        print(f"\nFinal result:")
        print(f"Best schedule: {best_fitness}")
        print(f"Execution time: {end_time - start_time}")
        print_schedule(best_solution, data)
        
        # Print final schedule