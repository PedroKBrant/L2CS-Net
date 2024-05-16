from l2cs import Pipeline
import cv2
import torch
import os 

gaze_pipeline = Pipeline(
    weights='models/L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu')  # or 'gpu'
)
input_folder = '/home/voxar/Desktop/pkb/GANonymization/lib/datasets/00_pkb_test'
#input_folder = '/home/voxar/Desktop/pkb/GANonymization/lib/datasets/CelebA/celeba/img/FaceSegmentation/test/'
output_file = 'pkb/experiments/teste.csv'

# List all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Sort the image files list
image_files.sort()

# Open the output file for writing
with open(output_file, 'w') as f_out:
    for image_file in image_files:
        # Get the image name without the extension
        img_name = os.path.splitext(image_file)[0].split('-')[-1]

        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, image_file))

        # Process frame and visualize
        try:
            result = gaze_pipeline.step(frame)
            bboxes_list = result.bboxes.tolist()  # Convert numpy array to Python list
            landmarks_list = result.landmarks.tolist()  # Convert numpy array to Python list
            # Write the result to the output file

            f_out.write(f"{img_name}, {result.pitch[0]}, {result.yaw[0]}, {bboxes_list[0]}, {landmarks_list[0][0]}, {landmarks_list[0][1]}, {landmarks_list[0][2]}, {landmarks_list[0][3]}, {landmarks_list[0][4]},{result.scores[0]}\n")
        except:
            print(f"no results for {image_file}")