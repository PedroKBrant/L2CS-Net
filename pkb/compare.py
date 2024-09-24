import csv
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os

class Gaze:
    def __init__(self, id, pitch, yaw):
        self.id = id
        self.pitch = pitch
        self.yaw = yaw

    def __repr__(self):
        return f"Gaze(id={self.id}, pitch={self.pitch}, yaw={self.yaw})"

class GazeCollection:
    def __init__(self):
        self.gazes = []

    def add_gaze(self, gaze):
        self.gazes.append(gaze)

    def calculate_mean_pitch(self):
        pitches = [gaze.pitch for gaze in self.gazes]
        return statistics.mean(pitches)

    def calculate_std_pitch(self):
        pitches = [gaze.pitch for gaze in self.gazes]
        return statistics.stdev(pitches)

    def calculate_mean_yaw(self):
        yaws = [gaze.yaw for gaze in self.gazes]
        return statistics.mean(yaws)

    def calculate_std_yaw(self):
        yaws = [gaze.yaw for gaze in self.gazes]
        return statistics.stdev(yaws)
    
    def get_max_pitch(self):
        pitches = [gaze.pitch for gaze in self.gazes]
        return max(pitches)

    def get_min_pitch(self):
        pitches = [gaze.pitch for gaze in self.gazes]
        return min(pitches)

    def get_max_yaw(self):
        yaws = [gaze.yaw for gaze in self.gazes]
        return max(yaws)

    def get_min_yaw(self):
        yaws = [gaze.yaw for gaze in self.gazes]
        return min(yaws)
    
    def print_statistics(self):
        mean_pitch = self.calculate_mean_pitch()
        std_pitch = self.calculate_std_pitch()
        mean_yaw = self.calculate_mean_yaw()
        std_yaw = self.calculate_std_yaw()
        max_pitch = self.get_max_pitch()
        min_pitch = self.get_min_pitch()
        max_yaw = self.get_max_yaw()
        min_yaw = self.get_min_yaw()

        print(f"Mean Pitch: {mean_pitch}")
        print(f"Standard Deviation of Pitch: {std_pitch}")
        print(f"Max Pitch: {max_pitch}")
        print(f"Min Pitch: {min_pitch}")
        print(f"Mean Yaw: {mean_yaw}")
        print(f"Standard Deviation of Yaw: {std_yaw}")
        print(f"Max Yaw: {max_yaw}")
        print(f"Min Yaw: {min_yaw}")

    def calculate_angular_errors(self, other_collection, visualize=False):
        errors = []
        # Create a dictionary for quick look-up of gazes by id in the other collection
        other_gazes_dict = {gaze.id: gaze for gaze in other_collection.gazes}
        for gaze in self.gazes:
            if gaze.id in other_gazes_dict:
                other_gaze = other_gazes_dict[gaze.id]
                pitch_error = gaze.pitch - other_gaze.pitch
                yaw_error = gaze.yaw - other_gaze.yaw
                errors.append((gaze.id, pitch_error, yaw_error))
        if visualize:
            for error in errors:
                print(f"ID: {error[0]}, Pitch Error: {error[1]}, Yaw Error: {error[2]}")
        return errors
    
    def plot_density_plots(self, other_collection):
        # Extract pitch and yaw values
        pitch_data_1 = [gaze.pitch for gaze in self.gazes]
        pitch_data_2 = [gaze.pitch for gaze in other_collection.gazes]
        yaw_data_1 = [gaze.yaw for gaze in self.gazes]
        yaw_data_2 = [gaze.yaw for gaze in other_collection.gazes]

        # Plot density plots for Pitch
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.kdeplot(pitch_data_1, label='Original', shade=True)
        sns.kdeplot(pitch_data_2, label='Anonymized', shade=True)
        plt.title('Density Plot of Pitch')
        plt.legend()

        # Plot density plots for Yaw
        plt.subplot(1, 2, 2)
        sns.kdeplot(yaw_data_1, label='Original', shade=True)
        sns.kdeplot(yaw_data_2, label='Anonymized', shade=True)
        plt.title('Density Plot of Yaw')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def calculate_mean_error(self, angular_errors):
        pitch_errors = [abs(error[1]) for error in angular_errors]
        yaw_errors = [abs(error[2]) for error in angular_errors]

        mean_pitch_error = round(statistics.mean(pitch_errors), 4)
        mean_yaw_error = round(statistics.mean(yaw_errors), 4)
        return mean_pitch_error, mean_yaw_error

    def plot_angular_error_box_plots(self, angular_errors, plot_flag=True):
        # Extract pitch and yaw errors
        pitch_errors = [abs(error[1]) for error in angular_errors]
        yaw_errors = [abs(error[2]) for error in angular_errors]

        # Calculate means and standard deviations
        mean_pitch_error = round(statistics.mean(pitch_errors), 4)
        std_pitch_error = round(statistics.stdev(pitch_errors), 4)
        mean_yaw_error = round(statistics.mean(yaw_errors), 4)
        std_yaw_error = round(statistics.stdev(yaw_errors), 4)

        # Print mean and standard deviation for pitch and yaw errors
        print(f"Mean Pitch Error: {mean_pitch_error}")
        #print(f"Standard Deviation of Pitch Error: {std_pitch_error}")
        print(f"Mean Yaw Error: {mean_yaw_error}")
        #print(f"Standard Deviation of Yaw Error: {std_yaw_error}")

        
        if(plot_flag):
            data = {
                'Pitch Error': pitch_errors,
                'Yaw Error': yaw_errors
            }

            df = pd.DataFrame(data)

            # Plot box plots for Pitch and Yaw Errors
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.boxplot(y='Pitch Error', data=df, showfliers=False)
            plt.title('Box Plot of Pitch Errors')

            plt.subplot(1, 2, 2)
            sns.boxplot(y='Yaw Error', data=df, showfliers=False)
            plt.title('Box Plot of Yaw Errors')

            plt.tight_layout()
            plt.show()
        
    
    def plot_jointplot(self, other_collection):
        # Extract pitch and yaw data from both collections
        pitch_data_1 = [gaze.pitch for gaze in self.gazes]
        yaw_data_1 = [gaze.yaw for gaze in self.gazes]
        pitch_data_2 = [gaze.pitch for gaze in other_collection.gazes]
        yaw_data_2 = [gaze.yaw for gaze in other_collection.gazes]

        # Create a DataFrame for the joint plot
        df1 = pd.DataFrame({
            'Pitch': pitch_data_1,
            'Yaw': yaw_data_1,
            'Dataset': ['CelebA'] * len(pitch_data_1)
        })
        df2 = pd.DataFrame({
            'Pitch': pitch_data_2,
            'Yaw': yaw_data_2,
            'Dataset': ['MetaGaze'] * len(pitch_data_2)
        })

        df = pd.concat([df1, df2])

        # Plot the joint plot with hue to differentiate datasets
        sns.jointplot(x='Pitch', y='Yaw', data=df, hue='Dataset', kind='scatter')
        plt.show()

def read_csv(filepath):
    invalid_values=0
    collection = GazeCollection()
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if(False):#('_' in row[0]):
                id = str(row[0]).split('_')[1].lstrip('0').replace('.png', '')
                if(len(id)==0):
                    id = 0
                id = int(id)
            else:
                id = str(row[0])
            pitch = float(row[1])
            yaw = float(row[2])
            if pitch > -5 and yaw > -5:
                collection.add_gaze(Gaze(id, pitch, yaw))#Filter value -10, withc means error
            else:
                invalid_values=invalid_values+1

    #print("invalid_values", round(invalid_values/6984, 3))
    return collection
'''
files_list = ['/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/BASELINE.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/Cel.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/Cel+MG.csv','/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/DP2.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/MESH_02.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/MESH_03.csv','/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/MG.csv', '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/MG+Cel.csv']
original = read_csv('/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/BASELINE.csv')

for file in files_list:
    print(file)
    anonymized = read_csv(file)
    #anonymized.print_statistics()
    angular_errors = original.calculate_angular_errors(anonymized, False)
    original.plot_angular_error_box_plots(angular_errors)
#original.plot_density_plots(anonymized_00)
# Plot jointplot for the original and anonymized data
#original.plot_jointplot(anonymized_00)

'''

plot = 'eyecloseness'

if plot=='camera':
    csv_dir_path = '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/splitted/camera'
    original = read_csv('/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/BASELINE.csv')
    csv_file_paths = glob.glob(os.path.join(csv_dir_path, '*.csv'))
    plot_list = []
    for file_path in csv_file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        #print('\n'+base_name+'\n_______________________')
        anonymized = read_csv(file_path)
        angular_errors = original.calculate_angular_errors(anonymized, False)
        mean_pitch_error, mean_yaw_error = original.calculate_mean_error(angular_errors)
        plot_list.append((base_name, mean_pitch_error, mean_yaw_error))

    plot_list.sort(key=lambda x: x[0])
    variations =2 
    # Generate the new fused list
    fused_list = [(plot_list[i][0], plot_list[i][1], plot_list[i][2], plot_list[i+1][1], plot_list[i+1][2]) 
                for i in range(0, len(plot_list), variations)]
    names = [x[0].rsplit('_', 1)[0] for x in fused_list] 
    pitch_errors = [x[1] for x in fused_list]
    yaw_errors = [x[2] for x in fused_list]
    pitch_errors2 = [x[3] for x in fused_list]
    yaw_errors2 = [x[4] for x in fused_list]
    x = np.arange(len(names))  
    width = 0.2  # Smaller width to accommodate more bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Adjust positions for bars1, bars2, bars3, and bars4
    bars1 = ax.bar(x - 1.5*width, pitch_errors, width, label='Pitch camera < 10', color='b')  # First group
    bars2 = ax.bar(x - 0.5*width, pitch_errors2, width, label='Pitch camera > 10', color='g')  # Third group
    bars3 = ax.bar(x + 0.5*width, yaw_errors, width, label='Yaw camera < 10', color='r')      # Second group
    bars4 = ax.bar(x + 1.5*width, yaw_errors2, width, label='Yaw camera > 10', color='y')      # Fourth group

    # Add labels and title
    ax.set_xlabel('Camera')
    ax.set_ylabel('Mean Error')
    ax.set_title('Pitch and Yaw Mean Errors by Camera')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend()

    # Set y-axis ticks at 0.05 intervals
    ax.set_yticks(np.arange(0, max(max(pitch_errors), max(yaw_errors)) + 0.05, 0.05))

    # Add grid to the plot with steps of 0.05
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plot
    plt.tight_layout()
    plt.show()

elif plot=='gaze':
    csv_dir_path = '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/splitted/gaze'
    original = read_csv('/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/BASELINE.csv')
    csv_file_paths = glob.glob(os.path.join(csv_dir_path, '*.csv'))
    plot_list = []
    for file_path in csv_file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        #print('\n'+base_name+'\n_______________________')
        anonymized = read_csv(file_path)
        angular_errors = original.calculate_angular_errors(anonymized, False)
        mean_pitch_error, mean_yaw_error = original.calculate_mean_error(angular_errors)
        plot_list.append((base_name, mean_pitch_error, mean_yaw_error))

    plot_list.sort(key=lambda x: x[0])
    variations =2 
    # Generate the new fused list
    fused_list = [(plot_list[i][0], plot_list[i][1], plot_list[i][2], plot_list[i+1][1], plot_list[i+1][2]) 
                for i in range(0, len(plot_list), variations)]
    names = [x[0].rsplit('_', 1)[0] for x in fused_list] 
    pitch_errors = [x[1] for x in fused_list]
    yaw_errors = [x[2] for x in fused_list]
    pitch_errors2 = [x[3] for x in fused_list]
    yaw_errors2 = [x[4] for x in fused_list]
    x = np.arange(len(names))  
    width = 0.2  # Smaller width to accommodate more bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Adjust positions for bars1, bars2, bars3, and bars4
    bars1 = ax.bar(x - 1.5*width, pitch_errors, width, label='Pitch gaze < 30', color='b')  # First group
    bars2 = ax.bar(x - 0.5*width, pitch_errors2, width, label='Pitch gaze > 30', color='g')  # Third group
    bars3 = ax.bar(x + 0.5*width, yaw_errors, width, label='Yaw gaze < 30', color='r')      # Second group
    bars4 = ax.bar(x + 1.5*width, yaw_errors2, width, label='Yaw gaze > 30', color='y')      # Fourth group

    # Add labels and title
    ax.set_xlabel('Gaze')
    ax.set_ylabel('Mean Error')
    ax.set_title('Pitch and Yaw Mean Errors by Gaze')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend()

    # Set y-axis ticks at 0.05 intervals
    ax.set_yticks(np.arange(0, max(max(pitch_errors), max(yaw_errors)) + 0.05, 0.05))

    # Add grid to the plot with steps of 0.05
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plot
    plt.tight_layout()
    plt.show()

elif plot=='fov':
    csv_dir_path = '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/splitted/fov'
    original = read_csv('/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/BASELINE.csv')
    csv_file_paths = glob.glob(os.path.join(csv_dir_path, '*.csv'))
    plot_list = []
    for file_path in csv_file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        #print('\n'+base_name+'\n_______________________')
        anonymized = read_csv(file_path)
        angular_errors = original.calculate_angular_errors(anonymized, False)
        mean_pitch_error, mean_yaw_error = original.calculate_mean_error(angular_errors)
        plot_list.append((base_name, mean_pitch_error, mean_yaw_error))

    plot_list.sort(key=lambda x: x[0])
    variations =3 
    # Generate the new fused list
    fused_list = [(plot_list[i][0], plot_list[i][1], plot_list[i][2], plot_list[i+1][1], plot_list[i+1][2], plot_list[i+2][1], plot_list[i+2][2]) 
                for i in range(0, len(plot_list), variations)]
    names = [x[0].rsplit('_', 1)[0] for x in fused_list] 
    pitch_errors = [x[1] for x in fused_list]
    yaw_errors = [x[2] for x in fused_list]
    pitch_errors2 = [x[3] for x in fused_list]
    yaw_errors2 = [x[4] for x in fused_list]
    pitch_errors3 = [x[5] for x in fused_list]
    yaw_errors3 = [x[6] for x in fused_list]
    x = np.arange(len(names))  
    width = 0.12  # Smaller width to accommodate more bars

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - 2.5 * width, pitch_errors, width, label='Pitch fov = 60', color='b')  # First group
    bars2 = ax.bar(x - 1.5 * width, pitch_errors2, width, label='Pitch fov = 90', color='g')  # Third group
    bars3 = ax.bar(x - 0.5 * width, pitch_errors3, width, label='Pitch fov = 120', color='c')  # Fifth group

    bars4 = ax.bar(x + 0.5 * width, yaw_errors, width, label='Yaw fov = 60', color='r')      # Second group
    bars5 = ax.bar(x + 1.5 * width, yaw_errors2, width, label='Yaw fov = 90', color='y')      # Fourth group
    bars6 = ax.bar(x + 2.5 * width, yaw_errors3, width, label='Yaw fov = 120', color='m')      # Sixth group

    # Add labels and title
    ax.set_xlabel('FoV')
    ax.set_ylabel('Mean Error')
    ax.set_title('Pitch and Yaw Mean Errors by FoV')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend()

    # Set y-axis ticks at 0.05 intervals
    ax.set_yticks(np.arange(0, max(max(pitch_errors), max(yaw_errors)) + 0.05, 0.05))

    # Add grid to the plot with steps of 0.05
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plot
    plt.tight_layout()
    plt.show()

elif plot=='eyecloseness':
    csv_dir_path = '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/splitted/eyecloseness'
    original = read_csv('/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/BASELINE.csv')
    csv_file_paths = glob.glob(os.path.join(csv_dir_path, '*.csv'))
    plot_list = []
    for file_path in csv_file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        #print('\n'+base_name+'\n_______________________')
        anonymized = read_csv(file_path)
        angular_errors = original.calculate_angular_errors(anonymized, False)
        mean_pitch_error, mean_yaw_error = original.calculate_mean_error(angular_errors)
        plot_list.append((base_name, mean_pitch_error, mean_yaw_error))

    plot_list.sort(key=lambda x: x[0])
    variations =4 
    # Generate the new fused list
    fused_list = [(plot_list[i][0], plot_list[i][1], plot_list[i][2], plot_list[i+1][1], plot_list[i+1][2], plot_list[i+2][1], plot_list[i+2][2], plot_list[i+3][1], plot_list[i+3][2]) 
                for i in range(0, len(plot_list), variations)]
    names = [x[0].rsplit('_', 1)[0] for x in fused_list] 
    pitch_errors = [x[1] for x in fused_list]
    yaw_errors = [x[2] for x in fused_list]
    pitch_errors2 = [x[3] for x in fused_list]
    yaw_errors2 = [x[4] for x in fused_list]
    pitch_errors3 = [x[5] for x in fused_list]
    yaw_errors3 = [x[6] for x in fused_list]
    pitch_errors4 = [x[7] for x in fused_list]
    yaw_errors4 = [x[8] for x in fused_list]
    x = np.arange(len(names))  
    width = 0.1  # Smaller width to accommodate more bars

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - 3.5 * width, pitch_errors, width, label='Pitch eye semi closed', color='b')  # First group
    bars3 = ax.bar(x - 2.5 * width, pitch_errors2, width, label='Pitch eye half closed ', color='g')  # Third group
    bars5 = ax.bar(x - 1.5 * width, pitch_errors3, width, label='Pitch eye opened', color='c')  # Fifth group
    bars7 = ax.bar(x - 0.5 * width, pitch_errors4, width, label='Pitch eye wide opened', color='k')  # Seventh group

    bars2 = ax.bar(x + 0.5 * width, yaw_errors, width, label='Yaw eye semi closed ', color='r')      # Second group
    bars4 = ax.bar(x + 1.5 * width, yaw_errors2, width, label='Yaw eye half closed ', color='y')      # Fourth group
    bars6 = ax.bar(x + 2.5 * width, yaw_errors3, width, label='Yaw eye opened', color='m')      # Sixth group
    bars8 = ax.bar(x + 3.5 * width, yaw_errors4, width, label='Yaw eye wide opened', color='0.6')      # Eighth group

    # Add labels and title
    ax.set_xlabel('eyecloseness')
    ax.set_ylabel('Mean Error')
    ax.set_title('Pitch and Yaw Mean Errors by eyecloseness')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend()

    # Set y-axis ticks at 0.05 intervals
    ax.set_yticks(np.arange(0, max(max(pitch_errors), max(yaw_errors)) + 0.05, 0.05))

    # Add grid to the plot with steps of 0.05
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plot
    plt.tight_layout()
    plt.show()
