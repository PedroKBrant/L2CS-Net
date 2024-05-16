import csv
import statistics

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
    
    def print_statistics(self):
        mean_pitch = self.calculate_mean_pitch()
        std_pitch = self.calculate_std_pitch()
        mean_yaw = self.calculate_mean_yaw()
        std_yaw = self.calculate_std_yaw()

        print(f"Mean Pitch: {mean_pitch}")
        print(f"Standard Deviation of Pitch: {std_pitch}")
        print(f"Mean Yaw: {mean_yaw}")
        print(f"Standard Deviation of Yaw: {std_yaw}") 

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

gaze_collection_1 = read_csv(file_path_1)
gaze_collection_2 = read_csv(file_path_2)

gaze_collection_1.print_statistics()

angular_errors = gaze_collection_1.calculate_angular_errors(gaze_collection_2, True)