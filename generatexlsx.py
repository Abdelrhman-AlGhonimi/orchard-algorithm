import pandas as pd
import re
import os
import glob

# First, try to find the surgical schedule file
def find_surgical_schedule_file():
    # Check in the current directory first
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_files = [
        os.path.join(current_dir, "surgical_schedule.txt"),
        os.path.join(current_dir, "orchard_results.txt")
    ]
    
    # Also check for any txt files that might contain surgical schedules
    txt_files = glob.glob(os.path.join(current_dir, "*.txt"))
    possible_files.extend(txt_files)
    
    # Try to find a file that contains surgical schedule data
    for file_path in possible_files:
        if os.path.exists(file_path):
            # Check if file contains surgical schedule data
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Surgical Schedule:" in content:
                    print(f"Found surgical schedule in: {file_path}")
                    return file_path
    
    print("Could not find a surgical schedule file. Please specify the correct path.")
    return None

# Parse the surgical schedule from text file
def parse_surgical_schedule(file_path):
    rows = []
    current_room = None
    in_schedule_section = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Detect the start of the schedule section
            if "Surgical Schedule:" in line or "Final Schedule:" in line:
                in_schedule_section = True
                continue
            
            if not in_schedule_section:
                continue
            
            # Detect room headers
            room_match = re.match(r"Room (\d+):", line)
            if room_match:
                current_room = int(room_match.group(1))
                continue
            
            # Skip header and separator lines
            if line.startswith("===") or line.startswith("---") or "Start" in line and "End" in line:
                continue
            
            # Parse operation data lines
            if line.startswith("|") and current_room is not None:
                parts = [part.strip() for part in line.split("|")]
                
                # Skip lines with insufficient data
                if len(parts) < 8:
                    continue
                
                # Try to extract operation data
                try:
                    room = int(parts[1]) if parts[1].strip().isdigit() else current_room
                    surgeon = int(parts[2]) if parts[2].strip().isdigit() else 0
                    patient = int(parts[3]) if parts[3].strip().isdigit() else 0
                    duration = int(parts[4]) if parts[4].strip().isdigit() else 0
                    emergency = parts[5].strip()
                    start_time = parts[6].strip()
                    end_time = parts[7].strip()
                    
                    rows.append({
                        "Room": room,
                        "Surgeon": surgeon,
                        "Patient": patient,
                        "Duration (min)": duration,
                        "Emergency": emergency,
                        "Start Time": start_time,
                        "End Time": end_time
                    })
                except (ValueError, IndexError) as e:
                    print(f"Skipping invalid line: {line}")
                    print(f"Error: {e}")
    
    return rows

# Main function
def main():
    # Find the surgical schedule file
    input_file = find_surgical_schedule_file()
    if not input_file:
        return
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exels")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, "Orchard_algorithm.xlsx")
    
    # Parse the surgical schedule
    rows = parse_surgical_schedule(input_file)
    
    if not rows:
        print("No valid surgical schedule data found in the file.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by Room and Start Time (convert to datetime first for proper sorting)
    df["Start Time Parsed"] = pd.to_datetime(df["Start Time"], format="%I:%M %p", errors="coerce")
    df = df.sort_values(by=["Room", "Start Time Parsed"])
    df = df.drop("Start Time Parsed", axis=1)
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Successfully created Excel file with {len(rows)} operations: {output_file}")

if __name__ == "__main__":
    main()
