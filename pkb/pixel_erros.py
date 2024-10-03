import csv
import glob
import os
import numpy as np

def read_csv_and_extract(filename):
    pitch = []
    yaw = []
    
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        
        for row in reader:
            # Extract columns 8, 10, 12, 14, 16 using slicing
            pitch_values = row[7:16:2]  # Columns 8, 10, 12, 14, 16
            # Extract columns 9, 11, 13, 15, 17 using slicing
            yaw_values = row[8:17:2]  # Columns 9, 11, 13, 15, 17
            
            # Check if the size of extracted pitch and yaw values is 5
            if len(pitch_values) == 5:
                pitch.append(pitch_values)
            else:
                pitch.append([0, 0, 0, 0, 0])
                
            if len(yaw_values) == 5:
                yaw.append(yaw_values)
            else:
                yaw.append([0, 0, 0, 0, 0])
    
    return pitch, yaw

def calculate_l2_nme(pitch1, pitch2, reference_distance):
    # Check if the two lists have the same length
    if len(pitch1) != len(pitch2):
        raise ValueError("The two pitch lists must have the same length.")
    
    total_l2_all = 0
    total_l2_eyes = 0
    total_l2_nose = 0
    total_l2_mouth = 0
    
    num_elements_all = 0
    num_elements_eyes = 0
    num_elements_nose = 0
    num_elements_mouth = 0

    # Iterate through each corresponding array in pitch1 and pitch2
    for p1_array, p2_array in zip(pitch1, pitch2):
        # Ensure each sub-array has the same length (size 5 in this case)
        if len(p1_array) != len(p2_array):
            raise ValueError("Each corresponding array in the two pitch lists must have the same length.")
        
        # Skip arrays that contain zeros
        if 0 in p1_array or 0 in p2_array:
            continue
        
        # Calculate L2 for all elements
        l2_all = np.linalg.norm(np.array(p1_array, dtype=float) - np.array(p2_array, dtype=float))
        total_l2_all += l2_all
        num_elements_all += 1
        
        # Calculate L2 for eyes (first two elements)
        l2_eyes = np.linalg.norm(np.array(p1_array[:2], dtype=float) - np.array(p2_array[:2], dtype=float))
        total_l2_eyes += l2_eyes
        num_elements_eyes += 1
        
        # Calculate L2 for nose (next two elements)
        l2_nose = np.linalg.norm(np.array(p1_array[2:4], dtype=float) - np.array(p2_array[2:4], dtype=float))
        total_l2_nose += l2_nose
        num_elements_nose += 1
        
        # Calculate L2 for mouth (fifth element)
        l2_mouth = np.linalg.norm(np.array([p1_array[4]], dtype=float) - np.array([p2_array[4]], dtype=float))
        total_l2_mouth += l2_mouth
        num_elements_mouth += 1
    
    # Calculate normalized mean error (NME)
    nme_all = (total_l2_all / num_elements_all) / reference_distance if num_elements_all > 0 else 0
    nme_eyes = (total_l2_eyes / num_elements_eyes) / reference_distance if num_elements_eyes > 0 else 0
    nme_nose = (total_l2_nose / num_elements_nose) / reference_distance if num_elements_nose > 0 else 0
    nme_mouth = (total_l2_mouth / num_elements_mouth) / reference_distance if num_elements_mouth > 0 else 0
    
    return nme_all, nme_eyes, nme_nose, nme_mouth

BASELINE_filename = '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/BASELINE.csv'
BASELINE_pitch, BASELINE_yaw = read_csv_and_extract(BASELINE_filename)
BASELINE_values = [
    [[float(x), float(y)] for x, y in zip(sublist1, sublist2)]
    for sublist1, sublist2 in zip(BASELINE_pitch, BASELINE_yaw)
]
csv_dir_path = '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/DMD/'
csv_file_paths = glob.glob(os.path.join(csv_dir_path, '*.csv')) 

for file_path in csv_file_paths:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    pitch, yaw = read_csv_and_extract(file_path)
    values = [
        [[float(x), float(y)] for x, y in zip(sublist1, sublist2)]
        for sublist1, sublist2 in zip(pitch, yaw)
    ]

    mae_all, mae_eyes, mae_nose, mae_mouth = calculate_l2_nme(values, BASELINE_values, 7.5)
    print(base_name)
    print(round(mae_all, 3))
    print(round(mae_eyes, 3))
    print(round(mae_nose, 3))
    print(round(mae_mouth, 3))

