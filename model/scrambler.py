"""
This script reads hand landmark data from a CSV file, scrambles the data, and writes the scrambled data to a new CSV file.
The scrambling is performed column-wise to maintain the structure of the data while shuffling the values.

Steps:
    1. Read the data from the input CSV file.
    2. Convert the data to a NumPy array and scramble it column-wise.
    3. Write the scrambled data to a new CSV file.

Dependencies:
    - CSV (csv)
    - NumPy (np)
"""

import csv
import numpy as np

input_file_path = "hand_landmarks.csv"
output_file_path = "scrambled_data.csv"

with open(input_file_path, 'r') as input_file:
    reader = csv.reader(input_file)
    data = list(reader)

header = data[0]
data = np.array(data[1:])

rows = data[:,:].astype(float)

scrambled_data = np.apply_along_axis(np.random.permutation, 0, rows)

with open(output_file_path, 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(header)
    writer.writerows(scrambled_data)

print(f'Scrambled data has been saved to {output_file_path}')
