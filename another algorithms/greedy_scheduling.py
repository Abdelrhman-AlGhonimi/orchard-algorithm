import prim
from datetime import datetime
import random
def greedy_scheduling(data):
    """Greedy scheduling algorithm that prioritizes emergency cases and shorter durations"""
    operations = data['operations']
    rooms = data['rooms']
    
    # Sort operations by priority: emergency first, then duration
    operations_sorted = sorted(operations, key=lambda x: (-x['emergency'], x['duration']))
    
    # Initialize solution
    solution = []
    
    # Initialize room start times
    room_start_times = {room: 480 for room in rooms}  # Start at 8:00 AM
    
    # Track room availability
    room_availability = {room: True for room in rooms}
    
    # Assign operations to rooms
    for op in operations_sorted:
        # Find the most suitable room
        suitable_rooms = []
        for room in rooms:
            start_time = room_start_times[room]
            if start_time + op['duration'] <= 960:  # End time must be before 4:00 PM
                suitable_rooms.append((room, start_time))
        
        if suitable_rooms:
            # If it's an emergency, prioritize the earliest available room
            if op['emergency']:
                room, start_time = min(suitable_rooms, key=lambda x: x[1])
            else:
                # For non-emergency, try to balance across rooms
                room, start_time = min(suitable_rooms, key=lambda x: room_start_times[x[0]])
            
            solution.append((room, start_time))
            room_start_times[room] = start_time + op['duration']
            
            # Mark room as unavailable if it's going to be busy for a long time
            if start_time + op['duration'] >= 900:  # If operation ends after 3:00 PM
                room_availability[room] = False
        
        # If no suitable rooms found, try to find the least busy room
        if not suitable_rooms:
            available_rooms = [r for r in rooms if room_availability[r]]
            if available_rooms:
                room = random.choice(available_rooms)
                start_time = room_start_times[room]
                solution.append((room, start_time))
                room_start_times[room] = start_time + op['duration']
    
    return solution

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
        
        # Generate greedy solution
        start_time = datetime.now()
        solution = greedy_scheduling(data)
        end_time = datetime.now()
        
        # Calculate fitness
        fitness = prim.objective_function(solution, data)
        
        # Save results
        output = []
        output.append("Greedy Scheduling Results:\n")
        output.append(f"Fitness: {fitness}\n")
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
        for i, (room, start_time) in enumerate(solution):
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
        prim.save_to_file('greedy_scheduling_schedule.txt', ''.join(output))
        
        print("\nResults saved to greedy_scheduling_schedule.txt")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
