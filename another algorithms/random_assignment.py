import random
import prim
from datetime import datetime

def random_assignment(data):
    """Randomly assign operations to rooms and times"""
    operations = data['operations']
    rooms = data['rooms']
    
    # Initialize solution
    solution = []
    
    for op in operations:
        # Randomly select a room
        room = random.choice(rooms)
        # Randomly select a start time between 8:00 AM and 4:00 PM
        start_time = random.randint(480, 960 - op['duration'])
        solution.append((room, start_time))
    
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
        
        # Generate random solution
        start_time = datetime.now()
        solution = random_assignment(data)
        end_time = datetime.now()
        
        # Calculate fitness
        fitness = prim.objective_function(solution, data)
        
        # Save results
        output = []
        output.append("Random Assignment Results:\n")
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
        prim.save_to_file('random_assignment_schedule.txt', ''.join(output))
        
        print("\nResults saved to random_assignment_schedule.txt")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
