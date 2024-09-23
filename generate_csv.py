from l2cs import Pipeline
import cv2
import torch
import os 

gaze_pipeline = Pipeline(
    weights='models/L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu')  # or 'gpu'
)
input_folder = '/home/voxar/Desktop/pkb/datasets/CelebA_test'
#input_folder = '/home/voxar/Desktop/pkb/GANonymization/lib/datasets/CelebA/celeba/img/FaceSegmentation/test/'
output_file = '/home/voxar/Desktop/pkb/CelebA.csv'

def generate_csv_metahuman(input_folder, output_file):
    # List all folders in the input folder
    MetaHumans = [name for name in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, name))]

    # Sort the image files list
    MetaHumans.sort()
    with open(output_file, 'w') as f_out:
        for MetaHuman in MetaHumans:
            MetaHuman_path = input_folder+'/'+MetaHuman
            images = [image_number for image_number in os.listdir(MetaHuman_path)]
            images.sort()
            for image in images:
                frame = cv2.imread(os.path.join(MetaHuman_path, image))
                try:
                    result = gaze_pipeline.step(frame)
                    bboxes_list = result.bboxes.tolist()  # Convert numpy array to Python list
                    landmarks_list = result.landmarks.tolist()  # Convert numpy array to Python list
                    # Write the result to the output file

                    f_out.write(f"{MetaHuman+'_'+image.split('.')[0]}, {result.pitch[0]:.4f}, {result.yaw[0]:.4f}, \
                                {bboxes_list[0][0]:.2f}, {bboxes_list[0][1]:.2f}, {bboxes_list[0][2]:.2f}, {bboxes_list[0][3]:.2f}, \
                                {landmarks_list[0][0][0]:.2f}, {landmarks_list[0][0][1]:.2f},\
                                {landmarks_list[0][1][0]:.2f}, {landmarks_list[0][1][1]:.2f},\
                                {landmarks_list[0][2][0]:.2f}, {landmarks_list[0][2][1]:.2f},\
                                {landmarks_list[0][3][0]:.2f}, {landmarks_list[0][3][1]:.2f},\
                                {landmarks_list[0][4][0]:.2f}, {landmarks_list[0][4][1]:.2f},\
                                {result.scores[0]:.4f}\n")
                except:
                    f_out.write(f"{MetaHuman+'_'+image}, {-10}, {-10}, {0}, {0}, {0}, {0}, {0}, {0},{0}\n")
                    print(f"no results for {MetaHuman+'_'+image}")

def generate_csv(input_folder, output_file):
    # List all folders in the input folder
    with open(output_file, 'w') as f_out:
        images = [image_number for image_number in os.listdir(input_folder)]
        images.sort()
        for image in images:
            frame = cv2.imread(os.path.join(input_folder, image))
            try:
                result = gaze_pipeline.step(frame)
                bboxes_list = result.bboxes.tolist()  # Convert numpy array to Python list
                landmarks_list = result.landmarks.tolist()  # Convert numpy array to Python list
                # Write the result to the output file

                f_out.write(f"{image.split('.')[0]}, {result.pitch[0]:.4f}, {result.yaw[0]:.4f}, \
                            {bboxes_list[0][0]:.2f}, {bboxes_list[0][1]:.2f}, {bboxes_list[0][2]:.2f}, {bboxes_list[0][3]:.2f}, \
                            {landmarks_list[0][0][0]:.2f}, {landmarks_list[0][0][1]:.2f},\
                            {landmarks_list[0][1][0]:.2f}, {landmarks_list[0][1][1]:.2f},\
                            {landmarks_list[0][2][0]:.2f}, {landmarks_list[0][2][1]:.2f},\
                            {landmarks_list[0][3][0]:.2f}, {landmarks_list[0][3][1]:.2f},\
                            {landmarks_list[0][4][0]:.2f}, {landmarks_list[0][4][1]:.2f},\
                            {result.scores[0]:.4f}\n")
            except:
                f_out.write(f"{image}, {-10}, {-10}, {0}, {0}, {0}, {0}, {0}, {0},{0}\n")
                print(f"no results for {image}")               
    '''
    exit()
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
    '''
#inputs_path = ['/home/voxar/Desktop/pkb/datasets/Experiments_MetaGaze/MG', '/home/voxar/Desktop/pkb/datasets/Experiments_MetaGaze/Cel+MG', '/home/voxar/Desktop/pkb/datasets/Experiments_MetaGaze/MG+Cel']#'/home/voxar/Desktop/pkb/datasets/Experiments_MetaGaze/MetaGaze_testset', '/home/voxar/Desktop/pkb/datasets/Experiments_MetaGaze/Cel', 
#inputs_path = [ '/home/voxar/Desktop/pkb/datasets/DMD/results/MG', '/home/voxar/Desktop/pkb/datasets/DMD/results/Cel+MG', '/home/voxar/Desktop/pkb/datasets/DMD/results/MG+Cel', '/home/voxar/Desktop/pkb/datasets/Mesh_results/02_DMD', '/home/voxar/Desktop/pkb/datasets/Mesh_results/02_MG', '/home/voxar/Desktop/pkb/datasets/Mesh_results/03_DMD', '/home/voxar/Desktop/pkb/datasets/Mesh_results/03_MG']
#'/home/voxar/Desktop/pkb/datasets/dp2_results/DMD', '/home/voxar/Desktop/pkb/datasets/dp2_results/MetaGaze', '/home/voxar/Desktop/pkb/datasets/DMD/results/Cel',
inputs_path = ['/home/voxar/Desktop/pkb/datasets/Mesh_results/03_DMD']
for path in inputs_path:
    print(path)  
    output_path = path +'.csv'
    print(output_path)
    generate_csv(path, output_path)