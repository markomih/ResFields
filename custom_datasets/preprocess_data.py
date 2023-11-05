
import argparse
from glob import glob
import os
from typing import List

import sys
import os
import shutil

from tqdm import tqdm
import open3d as o3d

from colmap2neus_utils import gen_poses, gen_cameras_dynamic


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

    parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
    parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
    parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
    parser.add_argument("--colmap_camera_params", default="", help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
    parser.add_argument("--single_camera", default=1, help="Use a single camera for all images (useful for videos).")


    parser.add_argument("--input_model_path", default='colmap_model', help="Path to the existing model for image registration.")
    parser.add_argument("--run_initial_colmap", action="store_true", help="Running colmap on the first frame of the dynamic pipeline")
    parser.add_argument("--run_transform_colmap2neus", action="store_true", help="Converting colmap format to json format")


    parser.add_argument("--run_make_dataset", action="store_true", help="Run make_dataset.py on the new images.")
    parser.add_argument("--begin_frame", type=int, default=0, help="Frame number to begin reading from.")
    parser.add_argument("--end_frame", type=int, default=-1, help="Frame number to stop reading at.")
    parser.add_argument("--by_step", type=int, default=33333, help="Step size between frames to read.")
    parser.add_argument("--extracted_images_folder", type=str, help="Folder path to save the extracted images.")

    args = parser.parse_args()
    return args


def do_system(args):
	print(f"==== running: {args}")
	err = os.system(args)
	if err:
		print("FATAL: command failed")
		sys.exit(err)


def run_initial_colmap(args):
    """
        Running colmap on the first frame of the dynamic pipeline
    """

    colmap_binary = "colmap"
    EXTRACTED_IMAGES_FOLDER = args.extracted_images_folder

    # find first frame
    first_frame = args.begin_frame

    # make a text file with list of images using the first frame
    image_list_path = 'new_images.txt'
    with open(image_list_path, "w") as f:
        # f.write(first_frame)
        list_of_candidates = glob(f'{EXTRACTED_IMAGES_FOLDER}/video_*_frame_{first_frame}.png')

        for candidate in list_of_candidates:
            f.write(candidate.replace(EXTRACTED_IMAGES_FOLDER + "/", ''))
            f.write('\n')


    # 1. colmap feature_extractor
    do_system(f"{colmap_binary} feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --database_path {args.colmap_db} --ImageReader.camera_params \"{args.colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera {args.single_camera} --image_path {EXTRACTED_IMAGES_FOLDER} --image_list_path {image_list_path}")

    # 2. colmap exhaustive_matcher
    do_system(f"{colmap_binary} {args.colmap_matcher}_matcher --database_path {args.colmap_db} --SiftMatching.guided_matching=true")

    # 3. Create directory and run colmap mapper
    # sparse_folder = "dynamic_sparse"
    sparse_folder = f'{args.input_model_path}_sparse'
    # sparse_text = "dynamic_text"
    sparse_text = f'{args.input_model_path}_text'
    
    do_system(f"mkdir -p {sparse_folder}")
    do_system(f"{colmap_binary} mapper --database_path {args.colmap_db} --image_path {EXTRACTED_IMAGES_FOLDER} --output_path {sparse_folder}")

    # 4. colmap bundle_adjuster
    do_system(f"{colmap_binary} bundle_adjuster --input_path {sparse_folder}/0 --output_path {sparse_folder}/0 --BundleAdjustment.refine_principal_point 1")

    # 5. Create directory and run colmap model_converter
    do_system(f"mkdir -p {sparse_text}")
    do_system(f"{colmap_binary} model_converter --input_path {sparse_folder}/0 --output_path {sparse_text} --output_type TXT")



def read_video_mkv(filename, 
                   begin_frame, 
                   end_frame, 
                   by_step, 
                   out_folder,
                   file_pattern="video_4_frame_"):
    # Create a reader object for the Azure Kinect MKV file
    reader = o3d.io.AzureKinectMKVReader()
    
    # Open the file using the reader
    if not reader.open(filename):
        print("Failed to open the file:", filename)
        return


    # Create the output folder (either it was just deleted, or it never existed)
    if not os.path.exists(out_folder):
        print('Creating folder: ', out_folder)
        os.makedirs(out_folder)
        os.makedirs(os.path.join(out_folder, "depth"))
        os.makedirs(os.path.join(out_folder, "rgb"))
        
    # Seek and read frames from begin_frame to end_frame with by_step step
    for ith_frame in tqdm(range(begin_frame, end_frame, by_step)):
        # Seek to the ith_frame
        reader.seek_timestamp(ith_frame)
        rgbd = reader.next_frame()

        i = 1
        while reader.is_eof() != True:
            if rgbd is None:
                reader.seek_timestamp(int(ith_frame)+i)
                rgbd = reader.next_frame()
                i = i + 1
            else:
                break

        # Get the RGBD image for the ith_frame
        # rgbd = reader.next_frame()
        if rgbd is None:
            if(reader.is_eof()):
                print("Reached end of file")
                break
            print(f"Failed to extract frame {ith_frame} from {filename}")
            reader.close()
            continue

        # Save the color image of the ith_frame
        color_filename = os.path.join(out_folder, f"rgb/{file_pattern}{ith_frame}.png")
        o3d.io.write_image(color_filename, rgbd.color)
        print(f"Saved rgb image of microsecond {ith_frame+i} to {color_filename}")

        # Save the depth image of the ith_frame
        if(hasattr(rgbd, 'depth')):
            depth_filename = os.path.join(out_folder, f"depth/{file_pattern}{ith_frame}.png")
            o3d.io.write_image(depth_filename, rgbd.depth)



    # Close the reader
    reader.close()

def run_transform_colmap2neus(args):
    # find from borders of the scene the number of frames
    # TODO DUSAN: assume it's 
    begin_frame = args.begin_frame
    end_frame = args.end_frame
    by_step = args.by_step
    args.neus_frames = len(range(begin_frame, end_frame, by_step))

    assert args.neus_frames == len(glob(args.extracted_images_folder + f"/cam_0/rgb/*.png")), "Number of frames in the folder doesn't match the number of frames in the video"

    

    # cp -r f'{args.input_model_path}_sparse' f'{args.input_model_path}_neus/sparse'
    do_system(f"mkdir -p {args.input_model_path}_neus")
    do_system(f"mkdir -p {args.input_model_path}_neus/sparse")
    do_system(f"cp -r {args.input_model_path}_sparse/* {args.input_model_path}_neus/sparse")


    gen_poses(f'{args.input_model_path}_neus', f'{args.colmap_matcher}_matcher')


    # Check if the files exist
    if not os.path.exists(f'{args.input_model_path}_neus/sparse_points_interest.ply'):
        print("sparse_points_interest.ply doesn't exist!")
        # return
    if not os.path.exists(f'{args.input_model_path}_neus/sparse_points.ply'):
        print("sparse_points.ply doesn't exist!")
        return
    
    # Prompt user
    choice = input("Do you want to manually create 'sparse_points_interest.ply'? If you say \"no\", we will copy paste sparse_points.ply to sparse_points_interest.ply (yes/no): ").strip().lower()

    if choice == 'yes':
        print("Please manually create the 'sparse_points_interest.ply'. Exiting.")
        return
    elif choice == 'no':
        shutil.copy(f'{args.input_model_path}_neus/sparse_points.ply', 
                    f'{args.input_model_path}_neus/sparse_points_interest.ply')
        print(f"'sparse_points.ply' has been copied to 'sparse_points_interest.ply'.")
    else:
        print("Invalid choice. Exiting.")


    gen_cameras_dynamic(args)


if __name__ == '__main__':

   

    parser = argparse.ArgumentParser(description="Run COLMAP with new images and optional image list.")
    args = parse_args()


    DATA_ROOT = "plane_dataset"

    static_video_paths = [
        f'{DATA_ROOT}/cam_0.mkv',
        f'{DATA_ROOT}/cam_1.mkv',
        f'{DATA_ROOT}/cam_2.mkv',
        f'{DATA_ROOT}/cam_3.mkv'
    ]    
    dynamic_video_paths = [
        # f'{DATA_ROOT}/3_video_with_charuco.mkv'
    ]

    video_config = {
        "static_videos": [f"{i}" for i in range(len(static_video_paths))],  # All videos except the last one are static for this example
        "dynamic_videos": [f"{i + len(static_video_paths)}" for i in range(len(dynamic_video_paths))], # Only the last video is dynamic for this example
    }

    args.video_config = video_config

    # TODO Dusan: add video extractor from which to extract frames from make_dataset.py
    if args.run_make_dataset:
        # For NeuS compatible format it goes a bit different and more convenient for later
        for video_idx in video_config["static_videos"]:
            read_video_mkv(static_video_paths[int(video_idx)], 
                        args.begin_frame, 
                        args.end_frame, 
                        args.by_step, 
                        args.extracted_images_folder + f"/cam_{video_idx}",
                        file_pattern=f"video_{video_idx}_frame_")
        for video_idx in video_config["dynamic_videos"]:
            read_video_mkv(dynamic_video_paths[int(video_idx) - len(static_video_paths)], 
                        args.begin_frame, 
                        args.end_frame, 
                        args.by_step, 
                        args.extracted_images_folder + f"/cam_{video_idx}",
                        file_pattern=f"video_{video_idx}_frame_")
            

            

    # add running first step
    if args.run_initial_colmap:
        choice = input("Do you need to run initial calibration step (i.e. with charuco)? (yes/no)").strip().lower()


        backup_images = args.extracted_images_folder
        backup_begin_frame = args.begin_frame
        
        if(choice == 'yes'):
            choice = input("Enter images folder: ").strip().lower()


            args.extracted_images_folder = choice
            # args.begin_frame = 1
            # args.begin_frame 
            images_list = glob(f'{args.extracted_images_folder}/*.png')
            args.begin_frame = int(images_list[0].split('/')[-1].split('frame_')[1].split('.')[0])


        run_initial_colmap(args)

        args.extracted_images_folder = backup_images
        args.begin_frame = backup_begin_frame



    if args.run_transform_colmap2neus:
        run_transform_colmap2neus(args)
