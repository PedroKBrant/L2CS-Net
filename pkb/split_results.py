import glob
import csv
import os

class Person:
    def __init__(self, id=None, fov=None, eye_closeness=None, eye_x=None, eye_y=None, camera_pitch=None, camera_yaw=None):
        self.id = str(id).zfill(4)  # Zero-pad the ID to 4 digits
        self.fov = fov
        self.eye_closeness = eye_closeness
        self.eye_x = eye_x
        self.eye_y = eye_y
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw

    @classmethod
    def from_csv(cls, csv_file):
        people = []
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file, delimiter=',')  # Assuming comma-separated CSV
            print("CSV Column Names:", reader.fieldnames)  # Print column names
            for row in reader:
                try:
                    person = cls(
                        id=str(row['id']).zfill(4),  # Zero-pad ID to 4 digits and cast to string
                        fov=int(row['fov']),
                        eye_closeness=float(row['eye_closeness']),
                        eye_x=float(row['eye_x']),
                        eye_y=float(row['eye_y']),
                        camera_pitch=float(row['camera_pitch']),
                        camera_yaw=float(row['camera_yaw'])
                    )
                    people.append(person)
                except KeyError as e:
                    print(f"KeyError: {e}. Check that the column names match exactly.")
                except ValueError as e:
                    print(f"ValueError: {e}. Check that the values in the CSV are of the correct type.")
        print(f"Loaded {len(people)} people from the CSV.")
        return people

    @classmethod
    def filter(cls, people, attribute, min_value, max_value):
        filtered_ids = []
        for person in people:
            attr_value = getattr(person, attribute, None)
            if attr_value is not None and min_value <= attr_value <= max_value:
                filtered_ids.append(person.id)  # IDs will be zero-padded strings
        return filtered_ids

    def __repr__(self):
        return (f"Person(id={self.id}, fov={self.fov}, eye_closeness={self.eye_closeness}, "
                f"eye_x={self.eye_x}, eye_y={self.eye_y}, camera_pitch={self.camera_pitch}, camera_yaw={self.camera_yaw})")


def process_csv(file_path, group, output_name):
    output_file = f'pkb/splitted/{output_name}.csv' 
    with open(file_path, mode='r') as file, open(output_file, mode='w', newline='') as output:
        csv_reader = csv.reader(file)
        csv_writer = csv.writer(output)

        for row in csv_reader:
            if row:  # Ensure the row is not empty
                first_column_value = row[0]
                start = first_column_value.index('_') + 1
                if '.' in first_column_value:
                    end = first_column_value.index('.')
                    extracted_value = first_column_value[start:end]
                else:
                    extracted_value = first_column_value[start:]
                    
                # Print the extracted value
                if extracted_value in group:
                    csv_writer.writerow(row)


csv_attributes_path = '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/MetaGaze_attributes.csv'
people = Person.from_csv(csv_attributes_path)



groups = {
    'fov60':  Person.filter(people, 'fov', 60, 60),
    'fov90':  Person.filter(people, 'fov', 90, 90),
    'fov120': Person.filter(people, 'fov', 120, 120),
    'eye_closeness-1':  Person.filter(people, 'eye_closeness', -1, -1),
    'eye_closeness0':   Person.filter(people, 'eye_closeness', 0, 0),
    'eye_closeness0.3': Person.filter(people, 'eye_closeness', 0.3, 0.3),
    'eye_closeness0.5': Person.filter(people, 'eye_closeness', 0.5, 0.5),
    'eye_closeness1':   Person.filter(people, 'eye_closeness', 1, 1),
    'gaze1': list(#union
        set(Person.filter(people, 'eye_x', -1, -1)) | 
        set(Person.filter(people, 'eye_x', 1, 1))  | 
        set(Person.filter(people, 'eye_y', -1, -1)) | 
        set(Person.filter(people, 'eye_y', 1, 1))
    ),
    'gaze0.5': list(#intersect 
        set(Person.filter(people, 'eye_x', -0.5, 0.5)) &
        set(Person.filter(people, 'eye_y', -0.5, 0.5))
    ),
    'camera20': list(#union
        set(Person.filter(people, 'camera_pitch', -20, -20)) | 
        set(Person.filter(people, 'camera_pitch', 20, 20))  | 
        set(Person.filter(people, 'camera_yaw', -20, -20)) | 
        set(Person.filter(people, 'camera_yaw', 20, 20))
    ),
    'camera10': list(#intersect 
        set(Person.filter(people, 'camera_pitch', -10, 10)) &
        set(Person.filter(people, 'camera_yaw', -10, 10))
    )


}

csv_dir_path = '/home/voxar/Desktop/pkb/L2CS-Net-1/pkb/msc/MetaGaze/'
csv_file_paths = glob.glob(os.path.join(csv_dir_path, '*.csv')) 

for file_path in csv_file_paths:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(base_name)
    for group_name, group_list in groups.items():
        output_name = f'{base_name}_{group_name}'
        process_csv(file_path, group_list, output_name)
