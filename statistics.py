import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from openpyxl import load_workbook

def generate_statistics():
    # Read data from Excel file
    df = pd.read_excel('Exels/Orchard_algorithm.xlsx')
    
    # Standardize column names and types
    df = df.rename(columns={
        'Room': 'room',
        'Surgeon': 'surgeon',
        'Patient': 'patient',
        'Duration (min)': 'duration',
        'Emergency': 'emergency',
        'Start Time': 'start_time',
        'End Time': 'end_time'
    })
    
    # Ensure emergency is boolean
    df['emergency'] = df['emergency'].str.strip().str.lower() == 'yes'
    
    # Convert start_time to minutes since start of day
    def time_to_minutes(time_str):
        if isinstance(time_str, str):
            parts = time_str.split()
            time_part = parts[0].split(':')
            hour = int(time_part[0])
            minute = int(time_part[1])
            period = parts[1]
            
            # Convert to 24-hour format
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
                
            return hour * 60 + minute
        return 0  # Return 0 if time_str is not in expected format
    
    df['start_minutes'] = df['start_time'].apply(time_to_minutes)
    
    # 1. Distribution of operations by time
    plt.figure(figsize=(12, 6))
    
    # Create time bins (every hour)
    time_bins = np.arange(480, 960 + 1, 60)  # 8:00 AM to 4:00 PM
    time_labels = [f"{h:02d}:00" for h in range(8, 17)]
    
    plt.hist(df['start_minutes'], bins=time_bins, edgecolor='black')
    plt.xticks(time_bins, time_labels, rotation=45)
    plt.xticks(time_bins, time_labels, rotation=45)
    plt.title('Distribution of Operations by Time')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Operations')
    plt.tight_layout()
    plt.savefig('operations_distribution.png')
    plt.close()
    
    # 2. Daily demand for operating rooms
    plt.figure(figsize=(10, 6))
    
    # Calculate total operation time per room
    room_times = df.groupby('room')['duration'].sum()
    
    plt.bar(room_times.index, room_times.values)
    plt.title('Daily Demand for Operating Rooms')
    plt.xlabel('Operating Room Number')
    plt.ylabel('Total Operation Time (minutes)')
    plt.xticks(room_times.index)
    plt.tight_layout()
    plt.savefig('room_demand.png')
    plt.close()
    
    # 3. Analysis of emergency operations by time
    plt.figure(figsize=(12, 6))
    
    # Separate emergency and non-emergency operations
    emergency_ops = df[df['emergency'] == True]
    non_emergency_ops = df[df['emergency'] == False]
    
    plt.hist([emergency_ops['start_minutes'], non_emergency_ops['start_minutes']], 
             bins=time_bins, 
             label=['Emergency', 'Non-Emergency'], 
             stacked=True, edgecolor='black')
    plt.xticks(time_bins, time_labels, rotation=45)
    plt.title('Emergency Operations Distribution by Time')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Operations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('emergency_distribution.png')
    plt.close()
    
    print("Statistical charts have been generated:")
    print("1. operations_distribution.png - Distribution of operations by time")
    print("2. room_demand.png - Daily demand for operating rooms")
    print("3. emergency_distribution.png - Emergency operations distribution")

if __name__ == "__main__":
    generate_statistics()
