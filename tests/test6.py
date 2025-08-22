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

def clean_data(data):
    cleaned_operations = []
    for op in data['operations']:
        if not isinstance(op.get('surgeon'), int) or op['surgeon'] < 0:
            continue
        if not isinstance(op.get('patient'), int) or op['patient'] < 0:
            continue
        if not isinstance(op.get('duration'), int) or op['duration'] <= 0:
            continue
        if op.get('emergency') not in [True, False]:
            continue
        if op['surgeon'] not in data['surgeons']:
            continue
        cleaned_operations.append(op)
    data['operations'] = cleaned_operations
    return data

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

    # Create schedules
    room_schedule = {room: [] for room in rooms}
    surgeon_schedule = {surgeon: [] for surgeon in data['surgeons']}

    # Build schedules
    for i, (room, start_time) in enumerate(solution):
        op = operations[i]
        duration = op['duration']
        end_time = start_time + duration

        # Room overlaps
        for existing_op in room_schedule[room]:
            if start_time < existing_op[1] and end_time > existing_op[0]:
                overlap += min(end_time, existing_op[1]) - max(start_time, existing_op[0])

        # Surgeon conflicts
        for existing_op in surgeon_schedule[op['surgeon']]:
            if start_time < existing_op[1] and end_time > existing_op[0]:
                surgeon_conflict += min(end_time, existing_op[1]) - max(start_time, existing_op[0])

        # Add to schedules
        room_schedule[room].append((start_time, end_time, op['emergency']))
        surgeon_schedule[op['surgeon']].append((start_time, end_time))
        room_usage += duration

    # Emergency waiting time & idle time
    for room in rooms:
        schedule = sorted(room_schedule[room])
        for i, (start, end, emergency) in enumerate(schedule):
            if emergency and i > 0:
                prev_end = schedule[i - 1][1]
                emergency_wait += max(0, start - prev_end)
            if i > 0:
                prev_end = schedule[i - 1][1]
                idle_time += max(0, start - prev_end)

    # Penalty weights
    emergency_weight = 5.0
    idle_weight = 2.0
    overlap_weight = 10.0
    usage_weight = 1.0
    surgeon_weight = 10.0

    score = (
        -emergency_weight * emergency_wait
        - idle_weight * idle_time
        - overlap_weight * overlap
        - usage_weight * (room_usage - (len(operations) * 30))
        - surgeon_weight * surgeon_conflict
    )

    return score

# Initialize a random solution
def initialize_seedling(data):
    """Create an initial solution for scheduling operations with priority improvement"""
    solution = []
    operations = data['operations']
    rooms = data['rooms']
    day_duration = 480  # 8 hours work = 480 minutes

    # Sort operations by priority: emergency first, then duration, then surgeon
    operations_sorted = sorted(operations,
                               key=lambda x: (-x['emergency'], -x['duration'], x['surgeon']))

    # Initialize start times for each room
    room_start_times = {room: 0 for room in rooms}
    room_usage = {room: 0 for room in rooms}  # Track usage for room selection
    surgeon_rooms = {surgeon: [] for surgeon in data['surgeons']}  # Track which rooms each surgeon used

    # Group operations by surgeon
    surgeon_operations = collections.defaultdict(list)
    for op in operations_sorted:
        surgeon_operations[op['surgeon']].append(op)

    # Process operations
    for surgeon, ops in surgeon_operations.items():
        # Try to keep the same room for the same surgeon
        if surgeon_rooms[surgeon]:
            preferred_room = random.choice(surgeon_rooms[surgeon])
        else:
            # Choose a room that's already less used
            available_rooms = sorted(
                [(r, room_usage[r]) for r in rooms],
                key=lambda x: x[1]
            )
            preferred_room = available_rooms[0][0]

        for op in ops:
            # Check if preferred room can fit this operation
            if room_usage[preferred_room] + op['duration'] <= day_duration:
                room = preferred_room
            else:
                # Find a room with enough time left
                available_rooms = sorted(
                    [(r, room_usage[r]) for r in rooms if room_usage[r] + op['duration'] <= day_duration],
                    key=lambda x: x[1]
                )
                if available_rooms:
                    room = available_rooms[0][0]
                else:
                    # Use least used room even if it exceeds the day limit
                    room = min(rooms, key=lambda r: room_usage[r])

            # Calculate start time (after last operation in the room)
            start_time = room_start_times[room]
            duration = op['duration']
            end_time = start_time + duration + 1  # Add 1 minute gap between operations

            # Update room schedule
            room_start_times[room] = end_time
            room_usage[room] += duration

            # Record the room used for this surgeon
            surgeon_rooms[surgeon].append(room)

            # Append to solution
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

def minutes_to_am_pm(minutes):
    total_minutes = int(minutes)
    hours, mins = divmod(total_minutes, 60)
    period = "AM" if hours < 12 else "PM"
    hours = hours % 12 or 12  # Handle midnight and noon correctly
    return f"{hours:02d}:{mins:02d} {period}"


def print_schedule(solution, data):
    """Print the schedule in a readable format using AM/PM format"""
    operations = data['operations']
    rooms = data['rooms']

    room_schedule = {room: [] for room in rooms}

    for i, (room, start_time) in enumerate(solution):
        op = operations[i]
        duration = op['duration']
        end_time = start_time + duration

        start_time_str = minutes_to_am_pm(start_time)
        end_time_str = minutes_to_am_pm(end_time)

        room_schedule[room].append({
            'surgeon': op['surgeon'],
            'patient': op['patient'],
            'duration': duration,
            'emergency': 'Yes' if op['emergency'] else 'No',
            'start_time': start_time_str,
            'end_time': end_time_str
        })

    print("\nSurgical Schedule:")
    print("========================================================================")
    print("| Room | Surgeon | Patient | Duration | Emergency | Start       | End         |")
    print("========================================================================")

    for room in rooms:
        print(f"\nRoom {room}:")
        print("-" * 74)
        for op in sorted(room_schedule[room], key=lambda x: x['start_time']):
            print(f"| {room:2d}   | {op['surgeon']:2d}     | {op['patient']:2d}      | {op['duration']:2d}       | {op['emergency']:3s}     | {op['start_time']} | {op['end_time']} |")
        print("-" * 74)

def save_to_file(filename, content):
    """Save content to a text file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error saving to file: {e}")

if __name__ == "__main__":
    try:
        # 1. قراءة البيانات من ملف XML
        data = read_xml_data('1_3PLInfo.xml')

        # 2. تنظيف البيانات بعد القراءة
        data = clean_data(data)

        print(f"\nData read successfully. Number of operations: {len(data['operations'])}")
        print("\nOperations data:")
        for op in data['operations']:
            print(f"Surgeon {op['surgeon']}: Patient {op['patient']}, Duration {op['duration']}, Emergency: {op['emergency']}")

        print(f"\nNumber of rooms: {data['num_rooms']}")
        print(f"Available rooms: {data['rooms']}")
        print(f"Number of surgeons: {len(data['surgeons'])}")

        # 3. التحقق من وجود بيانات صالحة
        if len(data['operations']) == 0:
            print("Error: No valid operations found after cleaning!")
            sys.exit(1)
        if len(data['rooms']) == 0:
            print("Error: No rooms available!")
            sys.exit(1)

        # 4. تشغيل الخوارزمية
        start_time = datetime.now()
        best_solution, best_fitness = orchard_algorithm(data)
        end_time = datetime.now()

        # 5. التحقق من وجود حل جيد
        if best_solution is not None:
            # إنشاء محتوى التقرير
            output = []
            output.append("Surgical Schedule Report\n")
            output.append("=" * 70 + "\n")
            output.append(f"Execution time: {end_time - start_time}\n")
            output.append(f"Final fitness score: {best_fitness}\n\n")

            output.append(f"Number of operations: {len(data['operations'])}\n")
            output.append(f"Available rooms: {data['rooms']}\n")
            output.append(f"Number of surgeons: {len(data['surgeons'])}\n\n")

            output.append("Operations Details:\n")
            for op in data['operations']:
                output.append(f"Surgeon {op['surgeon']}: Patient {op['patient']}, Duration {op['duration']}, Emergency: {op['emergency']}\n")
            output.append("\n")

            output.append("Surgical Schedule:\n")
            output.append("=" * 90 + "\n")
            output.append("| Room | Surgeon | Patient | Duration | Emergency | Start       | End         |\n")
            output.append("=" * 90 + "\n")

            # جدولة العمليات في الغرف
            operations = data['operations']
            rooms = data['rooms']
            room_schedule = {room: [] for room in rooms}

            for i, (room, start_time_op) in enumerate(best_solution):
                op = operations[i]
                duration = op['duration']
                end_time_op = start_time_op + duration

                # تحويل الوقت إلى AM/PM
                start_time_str = minutes_to_am_pm(start_time_op)
                end_time_str = minutes_to_am_pm(end_time_op)

                room_schedule[room].append({
                    'surgeon': op['surgeon'],
                    'patient': op['patient'],
                    'duration': duration,
                    'emergency': 'Yes' if op['emergency'] else 'No',
                    'start_time': start_time_str,
                    'end_time': end_time_str
                })

            # طباعة الجدول النهائي
            for room in rooms:
                output.append(f"\nRoom {room}:\n")
                output.append("-" * 90 + "\n")
                sorted_ops = sorted(room_schedule[room], key=lambda x: x['start_time'])
                for op in sorted_ops:
                    output.append(
                        f"| {room:2d}   | {op['surgeon']:2d}     | {op['patient']:2d}      | {op['duration']:2d}       | {op['emergency']:3s}     | {op['start_time']} | {op['end_time']} |\n"
                    )
                output.append("-" * 90 + "\n")

            # حفظ النتيجة في ملف
            save_to_file('surgical_schedule.txt', ''.join(output))

            # عرض النتيجة على الشاشة
            print("\nFinal result:")
            print(f"Best schedule fitness: {best_fitness}")
            print(f"Execution time: {end_time - start_time}")
            print_schedule(best_solution, data)

        else:
            print("No valid solution found after maximum iterations.")
            save_to_file('surgical_schedule.txt', "Error: No valid schedule could be generated.")

    except Exception as e:
        print(f"Error occurred: {e}")