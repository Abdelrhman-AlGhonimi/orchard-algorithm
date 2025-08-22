import xml.etree.ElementTree as ET
import collections
import random
import sys
import numpy as np
from datetime import datetime

# Constants
MIN_BUFFER_TIME = 1
DAY_START = 8 * 60
DAY_END = 16 * 60
DAY_MIDPOINT = (DAY_START + DAY_END) / 2
EMERGENCY_BUFFER_TIME = 30
URGENT_BUFFER = 60


def read_xml_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    operations = []
    surgeons = set()
    rooms = set()
    
    description = root.find('description').text
    lines = description.split('\n')
    
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
        
        if line.strip().startswith('Patient in Room'):
            try:
                room_num = line.split('Room')[1].split(':')[0].strip()
                if room_num.isdigit():
                    room = int(room_num)
                    rooms.add(room)
            except (ValueError, IndexError):
                continue
    
    for line in lines:
        if line.strip().startswith('Day') and 'Room' in line:
            try:
                parts = line.strip().split()
                day = int(parts[1])
                room = int(parts[3])
            except (ValueError, IndexError):
                continue
    
    rooms = sorted(list(rooms))
    
    return {
        'operations': operations,
        'surgeons': list(surgeons),
        'rooms': rooms,
        'num_rooms': len(rooms)
    }


def objective_function(solution, data):
    """Calculate schedule quality"""
    operations = data['operations']
    rooms = data['rooms']
    day_duration = 480  # 8 hours = 480 minutes

    penalties = {
        'emergency_wait': 0,
        'idle_time': 0,
        'overlap': 0,
        'surgeon_conflict': 0,
        'room_usage': 0,
        'buffer_violation': 0,
        'early_finish': 0  # Penalty for not using the full day
    }

    room_schedule = {room: [] for room in rooms}
    surgeon_schedule = {surgeon: [] for surgeon in data['surgeons']}

    for i, (room, start_time) in enumerate(solution):
        op = operations[i]
        duration = op['duration']
        end_time = start_time + duration

        for existing_op in room_schedule[room]:
            if start_time < existing_op[1] and end_time > existing_op[0]:
                penalties['overlap'] += min(end_time, existing_op[1]) - max(start_time, existing_op[0])
                
            buffer_violation = MIN_BUFFER_TIME - abs(start_time - existing_op[1])
            if 0 < abs(start_time - existing_op[1]) < MIN_BUFFER_TIME:
                penalties['buffer_violation'] += buffer_violation

        for existing_op in surgeon_schedule[op['surgeon']]:
            if start_time < existing_op[1] and end_time > existing_op[0]:
                penalties['surgeon_conflict'] += min(end_time, existing_op[1]) - max(start_time, existing_op[0])

        room_schedule[room].append((start_time, end_time, op['emergency']))
        surgeon_schedule[op['surgeon']].append((start_time, end_time))
        penalties['room_usage'] += duration

    for room in rooms:
        schedule = sorted(room_schedule[room])
        # Encourage using the full day: penalize if last operation ends before 4PM (480 min)
        if schedule:
            last_end = max(end for _, end, _ in schedule)
            penalties['early_finish'] += max(0, 480 - last_end)
        
        for i, (start_time, end_time, emergency) in enumerate(schedule):
            if emergency:
                if i > 0:
                    prev_end = schedule[i-1][1]
                    penalties['emergency_wait'] += max(0, start_time - prev_end)
        
        if len(schedule) > 1:
            for i in range(len(schedule) - 1):
                end1 = schedule[i][1]
                start2 = schedule[i+1][0]
                penalties['idle_time'] += max(0, start2 - end1)

    weights = {
        'emergency_wait': 5.0,
        'idle_time': 2.0, 
        'overlap': 20.0,
        'surgeon_conflict': 50.0,
        'room_usage': 1.0,
        'buffer_violation': 15.0,
        'early_finish': 3.0  # Weight for not using the full day
    }

    score = 0
    for penalty, value in penalties.items():
        score -= weights[penalty] * value

    return score


def initialize_seedling(data, reserve_afternoon_slots=True):
    solution = []
    operations = data['operations']
    rooms = data['rooms']
    day_duration = 480  # 8 hours work = 480 minutes
    
    global alert_system
    alert_system = InterferenceAlert()
    
    operations_sorted = sorted(operations, 
                             key=lambda x: (-x['emergency'], -x['duration'], x['surgeon']))
    
    room_start_times = {room: DAY_START for room in rooms}
    room_usage = {room: 0 for room in rooms}
    room_operations_count = {room: 0 for room in rooms}
    surgeon_rooms = {surgeon: [] for surgeon in data['surgeons']}
    
    scheduled_ops = []
    
    reserved_slots = {}
    if reserve_afternoon_slots:
        reserved_slots = {room: [] for room in rooms}
        afternoon_minutes = DAY_END - DAY_MIDPOINT
        slots_per_room = min(2, max(1, int(afternoon_minutes / 120)))
        
        for room in rooms:
            for i in range(slots_per_room):
                # Reserve slots at strategic points in the afternoon
                slot_start = DAY_MIDPOINT + (i * (afternoon_minutes / slots_per_room))
                reserved_slots[room].append((slot_start, slot_start + URGENT_BUFFER))
    
    # Group operations by priority first (emergency first), then by surgeon
    emergency_ops = [op for op in operations_sorted if op['emergency']]
    non_emergency_ops = [op for op in operations_sorted if not op['emergency']]
    
    # Process emergency operations first
    for op in emergency_ops:
        # Try to find the best room with the earliest possible start time
        best_room = None
        earliest_start = DAY_END
        
        for room in rooms:
            # Calculate potential start time
            start_time = room_start_times[room]
            
            # If this start time is earlier than our current best
            if start_time < earliest_start:
                # Check if this would create a surgeon conflict
                is_available, conflict_op = check_surgeon_availability(
                    op['surgeon'], start_time, start_time + op['duration'], scheduled_ops
                )
                
                if is_available:
                    best_room = room
                    earliest_start = start_time
                else:
                    # If there's a conflict, we could try scheduling after the conflict
                    conflict_end = conflict_op["start_minutes"] + conflict_op["duration"] + EMERGENCY_BUFFER_TIME
                    if conflict_end < earliest_start:
                        best_room = room
                        earliest_start = conflict_end
        
        if best_room is None:
            # If no room works, use the one with earliest start time and log an alert
            best_room = min(rooms, key=lambda r: room_start_times[r])
            earliest_start = room_start_times[best_room]
            alert_system.add_alert(
                "Surgeon Conflict", 
                f"Emergency operation for patient {op['patient']} has a surgeon conflict"
            )
        
        # Update room state
        room = best_room
        start_time = earliest_start
        end_time = start_time + op['duration']
        
        # Add appropriate buffer time for next operation
        room_start_times[room] = end_time + EMERGENCY_BUFFER_TIME
        room_usage[room] += op['duration']
        room_operations_count[room] += 1
        
        # Track surgeon's rooms
        if op['surgeon'] not in surgeon_rooms:
            surgeon_rooms[op['surgeon']] = []
        surgeon_rooms[op['surgeon']].append(room)
        
        # Add operation to solution
        solution.append((room, start_time))
        
        # Add to scheduled operations
        scheduled_ops.append({
            "surgeon": op['surgeon'],
            "patient": op['patient'],
            "room": room,
            "start_minutes": start_time,
            "duration": op['duration'],
            "emergency": op['emergency']
        })
    
    # Process non-emergency operations by surgeon
    surgeon_operations = {}
    for op in non_emergency_ops:
        if op['surgeon'] not in surgeon_operations:
            surgeon_operations[op['surgeon']] = []
        surgeon_operations[op['surgeon']].append(op)
    
    for surgeon, ops in surgeon_operations.items():
        # Try to keep operations for the same surgeon in the same room
        if surgeon_rooms[surgeon]:
            preferred_room = random.choice(surgeon_rooms[surgeon])
        else:
            preferred_room = random.choice(rooms)
        
        for op in ops:
            # Select the best room using load balancing criteria
            available_rooms = []
            for r in rooms:
                # Check if room has enough time for this operation
                if room_usage[r] + op['duration'] <= day_duration:
                    available_rooms.append(r)
            
            if not available_rooms:
                # If no room has enough space, use the room with the least usage
                room = min(rooms, key=lambda r: room_usage[r])
                alert_system.add_alert(
                    "Room Capacity Exceeded", 
                    f"Room {room} exceeds capacity with operation for patient {op['patient']}",
                    "Medium"
                )
            else:
                # Calculate a balanced score for each room
                room_scores = {}
                day_max_duration = DAY_END - DAY_START
                
                for r in available_rooms:
                    # Factors:
                    # 1. Free time (higher is better) - 30% weight
                    # 2. Operations count (lower is better) - 20% weight
                    # 3. Preferred room bonus - 20% weight
                    # 4. Room distribution evenness - 30% weight
                    free_time = day_max_duration - room_usage[r]
                    free_time_score = free_time / day_max_duration
                    
                    ops_count = room_operations_count[r]
                    max_ops = max(room_operations_count.values()) if any(room_operations_count.values()) else 1
                    ops_count_score = 1.0 - (ops_count / max_ops) if max_ops > 0 else 1.0
                    
                    preferred_bonus = 1.0 if r == preferred_room else 0.0
                    
                    # Room distribution score - assess how evenly rooms are being used
                    total_usage = sum(room_usage.values())
                    ideal_usage_per_room = total_usage / len(rooms) if total_usage > 0 else 0
                    room_deviation = abs(room_usage[r] - ideal_usage_per_room)
                    max_deviation = max([abs(usage - ideal_usage_per_room) for usage in room_usage.values()]) if total_usage > 0 else 1
                    distribution_score = 1.0 - (room_deviation / max_deviation) if max_deviation > 0 else 1.0
                    
                    # Combined weighted score
                    room_scores[r] = (0.3 * free_time_score) + (0.2 * ops_count_score) + \
                                    (0.2 * preferred_bonus) + (0.3 * distribution_score)
                
                # Choose room with highest score
                room = max(available_rooms, key=lambda r: room_scores[r])
            
            # Calculate start time, considering reserved slots and required buffer time
            start_time = room_start_times[room]
            duration = op['duration']
            end_time = start_time + duration
            
            # Avoid reserved slots for emergency operations
            if room in reserved_slots:
                for reserved_start, reserved_end in reserved_slots[room]:
                    # If operation overlaps with a reserved slot, move it
                    if (start_time < reserved_end) and (end_time > reserved_start):
                        # Try to place before the reserved slot
                        if start_time < reserved_start and (reserved_start - start_time) >= duration:
                            end_time = start_time + duration
                        else:
                            # Place after the reserved slot
                            start_time = reserved_end
                            end_time = start_time + duration
            
            # Check for surgeon conflicts with existing scheduled operations
            is_available, conflict_op = check_surgeon_availability(
                op['surgeon'], start_time, end_time, scheduled_ops
            )
            
            if not is_available:
                # Find the next available time for this surgeon
                available_start = conflict_op["start_minutes"] + conflict_op["duration"] + MIN_BUFFER_TIME
                start_time = available_start
                end_time = start_time + duration
                
                # Log an alert about the conflict
                alert_system.add_alert(
                    "Surgeon Conflict Resolved", 
                    f"Operation for patient {op['patient']} rescheduled due to surgeon conflict",
                    "Low"
                )
            
            # Check day limits
            if end_time > DAY_END:
                # Try to find another room or slot
                alternative_found = False
                
                for alt_room in rooms:
                    if alt_room != room and room_start_times[alt_room] + duration <= DAY_END:
                        room = alt_room
                        start_time = room_start_times[alt_room]
                        end_time = start_time + duration
                        alternative_found = True
                        break
                
                if not alternative_found:
                    # If no alternative, move to start of day and log alert
                    start_time = DAY_START
                    end_time = DAY_START + duration
                    alert_system.add_alert(
                        "Day Limit Exceeded", 
                        f"Operation for patient {op['patient']} scheduled at start of day due to time constraints",
                        "Medium"
                    )
            
            # Update room times with buffer
            room_start_times[room] = end_time + MIN_BUFFER_TIME
            room_usage[room] += duration
            room_operations_count[room] += 1
            
            # Track surgeon's rooms
            surgeon_rooms[surgeon].append(room)
            
            # Add operation to solution
            solution.append((room, start_time))
            
            # Add to scheduled operations
            scheduled_ops.append({
                "surgeon": op['surgeon'],
                "patient": op['patient'],
                "room": room,
                "start_minutes": start_time,
                "duration": op['duration'],
                "emergency": op['emergency']
            })
    
    # Perform a final check for interference between operations
    check_schedule_for_interference(solution, operations, rooms)
    
    return solution


def local_search(solution, data, gradual_improvement=True):
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


def graft(strong_seedling, medium_seedling, data):
    operations = data['operations']
    rooms = data['rooms']
    
    # Copy the strong seedling
    new_solution = strong_seedling[:]
    
    # Select operations from medium seedling to graft
    num_grafts = random.randint(1, len(medium_seedling) // 2)
    indices = random.sample(range(len(medium_seedling)), num_grafts)
    
    # Try to graft operations from medium seedling
    for idx in indices:
        room, start_time = medium_seedling[idx]
        op = operations[idx]
        duration = op['duration']
        surgeon_id = op['surgeon']
        
        # Try to find a valid time slot in the strong seedling
        valid = False
        for t in range(DAY_START, DAY_END - duration, 15):  # Check every 15 minutes within working hours
            valid = True
            
            # Check for room conflicts (including buffer time)
            for i, (other_room, other_start) in enumerate(new_solution):
                if i != idx and other_room == room:
                    other_duration = operations[i]['duration']
                    other_end = other_start + other_duration
                    
                    # Check overlap including buffer time
                    if ((t < other_end + MIN_BUFFER_TIME) and (t + duration + MIN_BUFFER_TIME > other_start)):
                        valid = False
                        break
            
            # Check for surgeon conflicts
            if valid:
                for i, (other_room, other_start) in enumerate(new_solution):
                    if i != idx and operations[i]['surgeon'] == surgeon_id:
                        other_duration = operations[i]['duration']
                        other_end = other_start + other_duration
                        
                        # Surgeon cannot be in two places at once
                        if (t < other_end and t + duration > other_start):
                            valid = False
                            break
            
            if valid:
                new_solution[idx] = (room, t)
                break
    
    return new_solution


def orchard_algorithm(data, N=50, GYN=20, alpha=0.5, beta=0.5, gamma=0.05,
                      num_strong_screening=20, num_weak_screening=20, num_grafting=10,
                      max_iterations=500, verbose=True):
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

def format_schedule(solution, data):
    operations = data['operations']
    rooms = data['rooms']
    
    # Create schedule for each room
    room_schedule = {room: [] for room in rooms}
    
    # Sort operations by start time
    for i, (room, start_time) in enumerate(solution):
        op = operations[i]
        duration = op['duration']
        end_time = start_time + duration
        
        # Convert minutes to hours and minutes with AM/PM format
        start_hour = start_time // 60
        start_minute = start_time % 60
        end_hour = end_time // 60
        end_minute = end_time % 60
        
        # Format time as HH:MM AM/PM
        start_period = "AM" if start_hour < 12 else "PM"
        end_period = "AM" if end_hour < 12 else "PM"
        
        # Convert to 12-hour format
        start_hour_12 = start_hour if start_hour <= 12 else start_hour - 12
        start_hour_12 = 12 if start_hour_12 == 0 else start_hour_12
        end_hour_12 = end_hour if end_hour <= 12 else end_hour - 12
        end_hour_12 = 12 if end_hour_12 == 0 else end_hour_12
        
        start_time_str = f"{start_hour_12:02d}:{start_minute:02d} {start_period}"
        end_time_str = f"{end_hour_12:02d}:{end_minute:02d} {end_period}"
        
        room_schedule[room].append({
            'surgeon': op['surgeon'],
            'patient': op['patient'],
            'duration': duration,
            'emergency': 'Yes' if op['emergency'] else 'No',
            'start_time': start_time_str,
            'end_time': end_time_str,
            'start_minutes': start_time  # Keep original minutes for sorting
        })
    
    # Build the schedule string
    schedule_str = "Surgical Schedule:\n"
    schedule_str += "==========================================================================\n"
    schedule_str += "| Room | Surgeon | Patient | Duration | Emergency | Start        | End          |\n"
    schedule_str += "==========================================================================\n"
    
    for room in rooms:
        schedule_str += f"\nRoom {room}:\n"
        schedule_str += "-" * 90 + "\n"
        # Sort by original start time in minutes
        for op in sorted(room_schedule[room], key=lambda x: x['start_minutes']):
            schedule_str += f"| {room:2d} | {op['surgeon']:7d} | {op['patient']:7d} | {op['duration']:8d} | {op['emergency']:9s} | {op['start_time']:12s} | {op['end_time']:12s} |\n"
        schedule_str += "-" * 90 + "\n"
    
    return schedule_str

def print_schedule(solution, data):
    print("\n" + format_schedule(solution, data))

def save_to_file(filename, content):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error saving to file: {e}")

class InterferenceAlert:
    def __init__(self):
        self.alerts = []
    
    def add_alert(self, alert_type, message, severity="Low"):
        self.alerts.append({
            "type": alert_type,
            "message": message,
            "severity": severity
        })

def check_surgeon_availability(surgeon, start_time, end_time, scheduled_ops):
    for op in scheduled_ops:
        if op["surgeon"] == surgeon and start_time < op["start_minutes"] + op["duration"] and end_time > op["start_minutes"]:
            return False, op
    return True, None

def check_schedule_for_interference(solution, operations, rooms):
    # Group operations by room
    room_operations = {room: [] for room in rooms}
    
    for i, (room, start_time) in enumerate(solution):
        op = operations[i]
        end_time = start_time + op['duration']
        room_operations[room].append({
            "index": i,
            "start": start_time,
            "end": end_time,
            "duration": op['duration'],
            "surgeon": op['surgeon'],
            "patient": op['patient'],
            "emergency": op['emergency']
        })
    
    # Check for interference in each room
    for room, ops in room_operations.items():
        # Sort by start time
        sorted_ops = sorted(ops, key=lambda x: x["start"])
        
        # Check for overlaps in the same room
        for i in range(len(sorted_ops) - 1):
            current = sorted_ops[i]
            next_op = sorted_ops[i+1]
            
            # Check if there's enough buffer time
            if next_op["start"] < current["end"] + MIN_BUFFER_TIME:
                buffer = next_op["start"] - current["end"]
                if buffer < 0:  # Actual overlap
                    alert_system.add_alert(
                        "Operation Overlap", 
                        f"Room {room}: Operations for patients {current['patient']} and {next_op['patient']} overlap by {abs(buffer)} minutes",
                        "High"
                    )
                elif buffer < MIN_BUFFER_TIME:  # Insufficient buffer
                    alert_system.add_alert(
                        "Insufficient Buffer", 
                        f"Room {room}: Only {buffer} minutes between operations for patients {current['patient']} and {next_op['patient']}",
                        "Medium"
                    )
    
    # Check for surgeon conflicts across all rooms
    surgeon_operations = {}
    for room, ops in room_operations.items():
        for op in ops:
            surgeon = op["surgeon"]
            if surgeon not in surgeon_operations:
                surgeon_operations[surgeon] = []
            surgeon_operations[surgeon].append({
                "room": room,
                "start": op["start"],
                "end": op["end"],
                "patient": op["patient"]
            })
    
    # Check for overlaps for each surgeon
    for surgeon, ops in surgeon_operations.items():
        # Sort by start time
        sorted_ops = sorted(ops, key=lambda x: x["start"])
        
        # Check for overlaps
        for i in range(len(sorted_ops) - 1):
            current = sorted_ops[i]
            next_op = sorted_ops[i+1]
            
            # Check if there's an overlap
            if next_op["start"] < current["end"]:
                overlap = min(current["end"], next_op["end"]) - next_op["start"]
                alert_system.add_alert(
                    "Surgeon Conflict", 
                    f"Surgeon {surgeon}: Operations in rooms {current['room']} and {next_op['room']} overlap by {overlap} minutes",
                    "High"
                )

if __name__ == "__main__":
    try:
        # Read data from XML file
        data = read_xml_data('1_3PLInfo.xml')
        print(f"\nData read successfully. Number of operations: {len(data['operations'])}")
        print("\nOperations data:")
        for op in data['operations']:
            print(f"Surgeon {op['surgeon']}: Patient {op['patient']}, Duration {op['duration']}, Emergency: {op['emergency']}")
        print(f"\nNumber of rooms: {data['num_rooms']}")
        print(f"Available rooms: {data['rooms']}")
        print(f"Number of surgeons: {len(data['surgeons'])}")
        
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
            
            # Add algorithm results
            output.append("\nAlgorithm results:\n")
            output.append(f"Final fitness: {best_fitness}\n")

            # Final results
            output.append("\nFinal result:\n")
            output.append(f"Best schedule: {best_fitness}\n")
            output.append(f"Execution time: {end_time - start_time}\n")
            
            # Schedule
            output.append("\nSurgical Schedule:\n")
            output.append("==========================================================================\n")
            output.append("| Room | Surgeon | Patient | Duration | Emergency | Start        | End          |\n")
            output.append("==========================================================================\n")
            
            operations = data['operations']
            rooms = data['rooms']
            room_schedule = {room: [] for room in rooms}
            
            # Sort operations by start time
            for i, (room, start_time) in enumerate(best_solution):
                op = operations[i]
                duration = op['duration']
                end_time = start_time + duration
                
                # Convert minutes to hours and minutes with AM/PM format
                start_hour = start_time // 60
                start_minute = start_time % 60
                end_hour = end_time // 60
                end_minute = end_time % 60
                
                # Format time as HH:MM AM/PM
                start_period = "AM" if start_hour < 12 else "PM"
                end_period = "AM" if end_hour < 12 else "PM"
                
                # Convert to 12-hour format
                start_hour_12 = start_hour if start_hour <= 12 else start_hour - 12
                start_hour_12 = 12 if start_hour_12 == 0 else start_hour_12
                end_hour_12 = end_hour if end_hour <= 12 else end_hour - 12
                end_hour_12 = 12 if end_hour_12 == 0 else end_hour_12
                
                start_time_str = f"{start_hour_12:02d}:{start_minute:02d} {start_period}"
                end_time_str = f"{end_hour_12:02d}:{end_minute:02d} {end_period}"
                
                room_schedule[room].append({
                    'surgeon': op['surgeon'],
                    'patient': op['patient'],
                    'duration': duration,
                    'emergency': 'Yes' if op['emergency'] else 'No',
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'start_minutes': start_time  # Keep original minutes for sorting
                })
            
            # Sort operations by start time in each room
            for room in rooms:
                room_schedule[room].sort(key=lambda x: x['start_minutes'])
            
            # Print schedule for each room
            for room in rooms:
                output.append(f"\nRoom {room}:\n")
                output.append("-" * 90 + "\n")
                for op in room_schedule[room]:
                    output.append(f"| {room:2d} | {op['surgeon']:7d} | {op['patient']:7d} | {op['duration']:8d} | {op['emergency']:9s} | {op['start_time']:12s} | {op['end_time']:12s} |\n")
                output.append("-" * 90 + "\n")
            
            # Save to file
            save_to_file('surgical_schedule.txt', ''.join(output))
            
            # Print final results to console
            print(f"\nFinal result:")
            print(f"Best schedule: {best_fitness}")
            print(f"Execution time: {end_time - start_time}")
            print_schedule(best_solution, data)
        else:
            print("No solution found")
    except Exception as e:
        print(f"Error: {e}")