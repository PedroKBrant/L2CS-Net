import csv
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

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
        if(visualize):
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
        sns.kdeplot(pitch_data_1, label='Collection 1', shade=True)
        sns.kdeplot(pitch_data_2, label='Collection 2', shade=True)
        plt.title('Density Plot of Pitch')
        plt.legend()

        # Plot density plots for Yaw
        plt.subplot(1, 2, 2)
        sns.kdeplot(yaw_data_1, label='Collection 1', shade=True)
        sns.kdeplot(yaw_data_2, label='Collection 2', shade=True)
        plt.title('Density Plot of Yaw')
        plt.legend()

        plt.tight_layout()
        plt.show()

def read_csv(filepath):
    collection = GazeCollection()
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            id = int(row[0])
            pitch = float(row[1])
            yaw = float(row[2])
            collection.add_gaze(Gaze(id, pitch, yaw))
    return collection

file_path_1 = 'pkb/experiments/original.csv'
file_path_2 = 'pkb/experiments/00_pkb_test.csv'

original = read_csv(file_path_1)
anonymized_00 = read_csv(file_path_2)

original.print_statistics()

#angular_errors = original.calculate_angular_errors(anonymized_00, True)

original.plot_density_plots(anonymized_00)