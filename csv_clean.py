import csv

# Define the file path
file_path = 'adult.test'  # Replace with your actual file path

# Read the data, remove periods, and write back to the same file
with open(file_path, 'r', newline='') as infile:
    reader = csv.reader(infile)
    rows = []
    
    # Iterate over each row in the input file
    for row in reader:
        if row:  # Check if the row is not empty
            # Check if the last value ends with a period and remove it
            if row[-1].endswith('.'):
                row[-1] = row[-1][:-1]
        
        # Add the cleaned row to the list
        rows.append(row)

# Write the cleaned data back to the same file
with open(file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print(f"Periods removed and data updated in {file_path}.")
