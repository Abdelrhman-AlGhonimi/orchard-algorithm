import pandas as pd
import re

# Define input and output paths
input_file = "D:/abdo1/VS Code/test/another algorithms/greedy_scheduling_schedule.txt"
output_file = "D:/abdo1/VS Code/test/Exels/greedy_scheduling_schedule.xlsx"

rows = []
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        # Look for lines that start with |
        if line.startswith("|") and line.count("|") >= 7:
            parts = [x.strip() for x in line.split("|")[1:-1]]
            if len(parts) == 7:
                room, surgeon, patient, duration, emergency, start, end = parts
                if parts[0].isdigit():
                    # Convert time to HH:MM AM/PM format
                    def convert_time(time_str):
                        # Split hours and minutes
                        hours, minutes = map(int, time_str.split(":"))
                        # Convert to 12-hour format
                        period = "AM" if hours < 12 else "PM"
                        if hours == 0:
                            hours = 12
                        elif hours > 12:
                            hours -= 12
                        return f"{hours:02d}:{minutes:02d} {period}"
                    
                    rows.append({
                        "Room": int(parts[0]),
                        "Surgeon": int(parts[1]),
                        "Patient": int(parts[2]),
                        "Duration (min)": int(parts[3]),
                        "Emergency": parts[4],
                        "Start Time": convert_time(parts[5]),
                        "End Time": convert_time(parts[6])
                    })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save as Excel
df.to_excel(output_file, index=False)
print(f"Successfully saved: {output_file}")
