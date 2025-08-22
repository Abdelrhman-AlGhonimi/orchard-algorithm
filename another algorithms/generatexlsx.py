import pandas as pd
import re

# افتح الملف النصي الذي يحتوي على الجدول (أو الصق النص مباشرة)
input_file = "D:/abdo1/VS Code/test/another algorithms/genetic_scheduling_schedule.txt"  # ضع هنا اسم الملف النصي
output_file = "D:/abdo1/VS Code/test/another algorithms/genetic_scheduling_schedule.xlsx"

rows = []
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        # نبحث عن الصفوف التي تبدأ بعلامة |
        if line.startswith("|") and line.count("|") >= 7:
            parts = [x.strip() for x in line.split("|")[1:-1]]
            if len(parts) == 7:
                room, surgeon, patient, duration, emergency, start, end = parts
                if parts[0].isdigit():
                    rows.append({
                        "Room": int(parts[0]),
                        "Surgeon": int(parts[1]),
                        "Patient": int(parts[2]),
                        "Duration": int(parts[3]),
        "Emergency": parts[4],
        "Start": parts[5],
        "End": parts[6]
    })


# تحويل البيانات إلى DataFrame
df = pd.DataFrame(rows)

# حفظها كملف Excel
df.to_excel(output_file, index=False)
print(f"تم حفظ الملف بنجاح باسم: {output_file}")
