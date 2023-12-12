
import argparse
from glob import glob
import os
import os
from tqdm import tqdm
import open3d as o3d
import numpy as np
from pyk4a import PyK4APlayback, ImageFormat, calibration, transformation
import json
import cv2
import torch
from sklearn.neighbors import NearestNeighbors
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from scipy import ndimage
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection
from PIL import Image
import time
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Use charuco calibration for dataset creation.")

    parser.add_argument("--downscale_factor", type=int, default=1, help="Downscale factor for the images.")
    parser.add_argument("--align_depth", type=int, default=1, help="Check if you want to keep depths from Kinect.")
    parser.add_argument("--run_calibration_finding", action="store_true", help="Running calibriation for overlapping pointclouds")
    parser.add_argument("--run_transform_charuco_calibration2neus", action="store_true", help="Converting calibration format to neus format")

    parser.add_argument("--run_make_dataset", action="store_true", help="Run make_dataset.py on the new images.")
    parser.add_argument("--src_dir", type=str, help="Folder path to save the extracted images.")
    parser.add_argument("--ignore_depth", type=int, default=0, help="Ignore depth images.")


    args = parser.parse_args()
    return args


def do_system(args):
	print(f"==== running: {args}")
	err = os.system(args)
	if err:
		print("FATAL: command failed")
		sys.exit(err)


# TODO Dusan: reformat below code
def col(A):
  return A.reshape((-1, 1))

def row(A):
    return A.reshape((1, -1))

def points_coord_trans(xyz_source_coord, trans_mtx):
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord

def projectPoints(v, cam):
    v = v.reshape((-1, 3)).copy()
    return cv2.projectPoints(v, np.asarray([[0.0,0.0,0.0]]), np.asarray([0.0,0.0,0.0]), np.asarray(cam['camera_mtx']),
                            np.asarray(cam['k']))[0].squeeze()

def get_valid_idx(points_color_coord, color_cam, TH=1e-2):
    # 3D points --> 2D coordinates in color image
    uvs = projectPoints(points_color_coord, color_cam)  # [n_depth_points, 2]
    uvs = np.round(uvs).astype(int)
    valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)  # [n_depth_points], true/false
    valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
    valid_idx = np.logical_and(valid_x, valid_y)  # [n_depth_points], true/false
    valid_idx = np.logical_and(valid_idx, points_color_coord[:, 2] > TH)
    uvs = uvs[valid_idx == True]  # valid 2d coords in color img of 3d depth points
    # todo: use human mask here if mask in color img space
    return valid_idx, uvs

def unproject_depth_image(depth_image, cam):
    us = np.arange(depth_image.size) % depth_image.shape[1]  # (217088,)  [0,1,2,...,640, ..., 0,1,2,...,640]
    vs = np.arange(depth_image.size) // depth_image.shape[1]  # (217088,)  [0,0,...,0, ..., 576,576,...,576]
    ds = depth_image.ravel()  # (217088,) return flatten depth_image (still the same memory, not a copy)
    uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)  # [576*640, 3]

    # undistort depth map
    xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()), # intrinsic: Distortion coefficients (1x5): k   Camera matrix (3x3): camera_mtx
                                                np.asarray(cam['camera_mtx']), np.asarray(cam['k']))  # [217088, 1, 2]  camera_mtx (3x3): [f_x, 0, c_x, 0, f_y, c_y, 0,0,0 ]

    # unproject to 3d points in depth cam coord
    xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), col(uvd[:, 2])))  # [217088, 3]
    xyz_camera_space[:, :2] *= col(xyz_camera_space[:, 2])  # scale x,y by z, --> 3d coordinates in depth camera coordinate

    return xyz_camera_space   # [576*640, 3]
# TODO DUsan: reformat code above

def get_aligned_rgbd(depth_img_path, rgb_img_path, depth_cam, rgb_cam):
    depth_image = cv2.imread(depth_img_path, flags=-1).astype(float) #.astype(float)
    
    print(f' inside get_aligned, before calling , depth_image max {depth_image.max()},  shape {depth_image.shape}')
    # downsample depth image by 4 times
    rgb_image = cv2.cvtColor(cv2.imread(rgb_img_path, flags=-1), cv2.COLOR_BGR2RGB)
    # rgb_image = cv2.imread(rgb_img_path, flags=-1)
    depth_image = depth_image.astype(float) * 0.001
    # depth_image = depth_image.astype(float)
    pcl = unproject_depth_image(depth_image, depth_cam)
    default_color = [1.00, 0.75, 0.80]
    kinect_colors_main = np.tile(default_color, [pcl.shape[0], 1])
    kinect_points_color_coord = points_coord_trans(pcl, np.asarray(depth_cam['ext_depth2color']))
    # kinect_points_color_coord = points_coord_trans(pcl, depth2color_transform)
    kinect_valid_idx, kinect_uvs = get_valid_idx(kinect_points_color_coord, rgb_cam)
    kinect_points_color_coord = kinect_points_color_coord[kinect_valid_idx]
    kinect_colors_main[kinect_valid_idx == True, :3] = rgb_image[kinect_uvs[:, 1], kinect_uvs[:, 0]]
    kinect_colors_main = kinect_colors_main[kinect_valid_idx]
    kinect_colors_main /= 255
    pcl = pcl[kinect_valid_idx]

    # depth_image_reprojected = np.zeros([1080, 1920])
    depth_image_reprojected = np.zeros([rgb_image.shape[0], rgb_image.shape[1]])
    depth_image_reprojected[(kinect_uvs[:, 1], kinect_uvs[:, 0])] = kinect_points_color_coord[:, 2]

    return rgb_image, depth_image_reprojected, pcl


def image_to_cloud(img_c, img_d, rgb_cam):

  f_x, f_y = rgb_cam['f'][0], rgb_cam['f'][1]
  c_x, c_y = rgb_cam['c'][0], rgb_cam['c'][1]

  us = np.arange(img_d.size) % img_d.shape[1]
  vs = np.arange(img_d.size) // img_d.shape[1]
  ds = img_d.ravel()  # (217088,) return flatten depth_image (still the same memory, not a copy)
  uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)

  xyz = uvd.copy()
  xyz[:, 0] = ((xyz[:, 0]-c_x)*xyz[:, 2]) / f_x
  xyz[:, 1] = ((xyz[:, 1]-c_y)*xyz[:, 2]) / f_y

  rgb =  img_c.reshape([-1, 3]) / 255

  valid_depth = uvd[:,2]!=0
  rgb = rgb[valid_depth]
  xyz = xyz[valid_depth]
  uvd = uvd[valid_depth]

  return xyz, rgb, uvd

def smooth_edges(image, mask, sigma=4, erosion_iterations=1
                 ):
    # Load the image and the mask
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    # Define a kernel for the erosion operation
    kernel = np.ones((3,3), np.uint8)
    
    # Erode the mask to shrink it

    # eroded_mask = cv2.erode(mask_input, kernel, iterations=erosion_iterations)
    if(mask.dtype == bool):
        mask_input = mask.astype(np.uint8)  * 255
    eroded_mask = cv2.erode(mask_input, kernel, iterations=erosion_iterations)
    # Apply Gaussian blur to the mask
    smoothed_mask = cv2.GaussianBlur(eroded_mask, (0,0), sigma)

    # Apply smoothed_mask to image
    smoothed_image = image.copy()
    
    # print(np.unique(smoothed_mask))
    for i in range(3):
        smoothed_image[:, :, i] = image[:, :, i] * (smoothed_mask > 0)

    smoothed_image = smoothed_image[:,:,:3]

    return smoothed_image, smoothed_mask

def remove_green_spill_from_edges(img, mask, threshold=0.3, spill_factor=0.5):

    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # threshold = 1.5
    spill_mask = (green_channel > 1.2 * blue_channel) & (green_channel > 1.2 * red_channel) 
    

    # if(spill_mask.sum() > 10000):
        # result_image = np.zeros_like(img)
        # result_image[spill_mask] = img[spill_mask]
        # Image.fromarray(result_image).show()


    
    for i in range(3):
        img[spill_mask, i] = 0



    # # Smooth out edges
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    # # Define the kernel size for morphological operations
    # kernel_size = 3  # Adjust the kernel size as needed
    # # Optionally, you can apply Gaussian smoothing to further smoothen the edges
    # edge_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0) == 0

    # img[edge_mask, :] = 0

    return img

# def get_cloud(depth_image_path, rgb_image_path, cam_id, depth_cam, rgb_cam, mask=None):
def get_cloud(depth_image_path, rgb_image_path, cam_id, depth_cam, rgb_cam, mask=None, do_aligning=True, visualize_markers=False):

    
    if(do_aligning):
        img_c, img_d, pcl = get_aligned_rgbd(depth_image_path,
                                rgb_image_path,
                                # depth_cams[cam_id],
                                depth_cam,
                                # rgb_cams[cam_id])
                                rgb_cam)
                                # depth_cam['ext_depth2color'])
    else:
        img_c = cv2.cvtColor(cv2.imread(rgb_image_path, flags=-1), cv2.COLOR_BGR2RGB)
        img_d = cv2.imread(depth_image_path, flags=-1).astype(float) * 0.001

    if mask is not None:
        img_c[mask==0] = 0
        img_d[mask==0] = 0

    # xyz, rgb, uvd = image_to_cloud(img_c, img_d, rgb_cams[cam_id])
    xyz, rgb, uvd = image_to_cloud(img_c, img_d, rgb_cam)

    kpts, ids = detect_aruco_markers(rgb_image_path,visualize=visualize_markers)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(uvd[:, 0:2])
    kpts_3d = []
    for ix in range(0, len(kpts)):
        distances, indices = nbrs.kneighbors(kpts[ix][0])
        kpts_3d.append(xyz[indices].reshape([-1, 3]))

    return xyz, rgb, [kpts_3d, ids], [kpts, ids]

def detect_aruco_markers(img_path, visualize=True):
    # dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # parameters = cv2.aruco.DetectorParameters_create()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    # plot corners and ids on frame
    # Draw keypoints on the frame
    if(visualize):
        if ids is not None:
            for i in range(len(ids)):
                # Draw a bounding box around the detected marker
                cv2.aruco.drawDetectedMarkers(frame, corners)
                # Draw a text label with the ArUco ID at the top-left corner of the marker
                cv2.putText(frame, str(ids[i][0]), (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert the frame to a PIL Image
        Image.fromarray(frame).show()
    

    return corners, ids

def get_charuco_transform(keypoints):

  K = keypoints
  center = torch.mean(K, axis=0)
  K_centered = K - center
  _, _, R = torch.linalg.svd(K_centered)


def find_common_keypoints(kpts_list):
    # Unpack corners and ids from each keypoints in the list
    corners = [kpts[0] for kpts in kpts_list]
    ids = [list(kpts[1]) for kpts in kpts_list]

    common_keypoints = []
    # Iterate over ids in the first list (assuming all lists are of equal length)
    for ix0 in range(len(ids[0])):
        try:
            # Find the index of the current id in the other lists
            indexes = [ids_list.index(ids[0][ix0]) for ids_list in ids[1:]]
            # Gather corners corresponding to the common id from all lists
            common_corners = [corners[i][index] for i, index in enumerate([ix0] + indexes)]
            # Concatenate corners horizontally
            common_keypoints.append(np.concatenate(common_corners, axis=1))
        except ValueError:
            # Skip if the id is not found in any of the lists
            continue

    # Concatenate all common keypoints vertically
    if common_keypoints:
        common_keypoints = np.concatenate(common_keypoints, axis=0)
    else:
        common_keypoints = np.array([])

    return common_keypoints

def get_torch_cameras(rgb_cams, R, T, img_height=1080, img_width=1920):

  cameras = []
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  for ix in range(0, len(rgb_cams)):

    camera_matrix =  torch.Tensor(np.concatenate(rgb_cams[ix]['camera_mtx'], 0).reshape([1, 3, 3]))
    image_size = torch.as_tensor([[img_height, img_width]])
    camera = cameras_from_opencv_projection(torch.eye(3).unsqueeze(0),
                                            torch.zeros(3).reshape([1,  3]),
                                            camera_matrix, image_size=image_size)

    if ix != 0:
      camera.R = (axis_angle_to_matrix(R[ix]).T @ camera.R[0].to(device)).unsqueeze(0)
      camera.T = (-T[ix] @ camera.R[0]).unsqueeze(0)

    cameras.append(camera)

  return cameras

def run_calibration_finding(args, calibrations_folder):
    # rgb_images_path = os.listdir(args.src_dir)
    rgb_images_path = sorted(glob(args.src_dir + "/cam_*/rgb/*.png"))
    depth_images_path = sorted(glob(args.src_dir + "/cam_*/depth/*.png"))

    

    depth_cams = []
    rgb_cams = []

    for i in range(len(glob(args.src_dir + "/cam_*/"))):
        with open(f'{calibrations_folder}/cam_{i}/rgb_camera.json') as f:
            rgb_cams.append(json.load(f))
        with open(f'{calibrations_folder}/cam_{i}/depth_camera.json') as f:
            depth_cams.append(json.load(f))

    # if(args.downscale_factor != 1):

    #     # load and downscale rgb_images_path and depth_images_path by downscale_factor
    #     rgb_images = []
    #     depth_images = []

    #     for i in range(len(rgb_images_path)):

    #         rgb_image = cv2.imread(rgb_images_path[i], flags=-1)
    #         depth_image = cv2.imread(depth_images_path[i], flags=-1)

    #         rgb_image_downscaled = cv2.resize(rgb_image, (rgb_image.shape[1] // args.downscale_factor, rgb_image.shape[0] // args.downscale_factor))
    #         depth_image_downscaled = cv2.resize(depth_image, (depth_image.shape[1] // args.downscale_factor, depth_image.shape[0] // args.downscale_factor))

    #         rgb_images.append(rgb_image_downscaled)
    #         depth_images.append(depth_image_downscaled)

    #     # now, copy the full calibration_folder folder structure to new folder, and save downsampled images there
    #     if not os.path.exists(f'{args.src_dir}_downscaled_{args.downscale_factor}'):
    #         os.makedirs(f'{args.src_dir}_downscaled_{args.downscale_factor}')
    #     # copy calibration_folder to calibration_folder_downscaled
    #     os.system(f'cp -r {args.src_dir}/* {args.src_dir}_downscaled_{args.downscale_factor}/')
    #     for i in range(len(rgb_images_path)):
    #         cv2.imwrite(f'{args.src_dir}_downscaled_{args.downscale_factor}/cam_{i}/rgb/{rgb_images_path[i].split("/")[-1]}', rgb_images[i])
    #         cv2.imwrite(f'{args.src_dir}_downscaled_{args.downscale_factor}/cam_{i}/depth/{depth_images_path[i].split("/")[-1]}', depth_images[i])



    #     # also, downscale camera intrinsics and extrinsics
    #     keys_to_downscale = ['f', 'c', 'camera_mtx']
    #     for i in range(len(rgb_cams)):
    #         for key in keys_to_downscale:

    #             if(key == 'camera_mtx'):
    #                 rgb_cams[i][key] = np.array(rgb_cams[i][key]).reshape([3,3])
    #                 depth_cams[i][key] = np.array(depth_cams[i][key]).reshape([3,3])

    #                 rgb_cams[i][key][:2, :] = rgb_cams[i][key][:2, :] / args.downscale_factor
    #                 depth_cams[i][key][:2, :] = depth_cams[i][key][:2, :] / args.downscale_factor

    #             rgb_cams[i][key] = [item / args.downscale_factor for item in rgb_cams[i][key]]
    #             depth_cams[i][key] = [item / args.downscale_factor for item in depth_cams[i][key]]


    #     # Rename rgb and depth images path
    #     rgb_images_path = [rgb_image_path.replace(f'{args.src_dir}', f'{args.src_dir}_downscaled_{args.downscale_factor}') for rgb_image_path in rgb_images_path]
    #     depth_images_path = [depth_image_path.replace(f'{args.src_dir}', f'{args.src_dir}_downscaled_{args.downscale_factor}') for depth_image_path in depth_images_path]

    #     # rename args.src_dir 
    #     args.src_dir = f'{args.src_dir}_downscaled_{args.downscale_factor}'


    # check if common_keypoints exist
    xyzs_list = []
    rgbs_list = []

    if os.path.exists(f'{calibrations_folder}/common_keypoints.pt'):
        print('Found common_keypoints.pt, skipping optimization')
        common_keypoints = torch.load(f'{calibrations_folder}/common_keypoints.pt', map_location=torch.device('cpu'))

        for i in range(len(rgb_images_path)):
            xyzs_list.append(torch.load(f'{calibrations_folder}/xyzs_list_{i}.pt'))
            rgbs_list.append(torch.load(f'{calibrations_folder}/xyzs_list_{i}.pt'))

    else:        
        kpts_3d_list = []
        kpts_2d_list = []
        for i in range(len(rgb_images_path)):
            xyzs, rgbs, kpts_3d, kpts_2d = get_cloud(depth_images_path[i], rgb_images_path[i], i, depth_cams[i], rgb_cams[i], do_aligning=args.align_depth, visualize_markers=True)
            xyzs_list.append(xyzs)
            rgbs_list.append(rgbs)
            kpts_3d_list.append(kpts_3d)
            kpts_2d_list.append(kpts_2d)

        common_keypoints = find_common_keypoints(kpts_3d_list)
        common_keypoints_2d = find_common_keypoints(kpts_2d_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_keypoints = torch.Tensor(common_keypoints).to(device)

    # save common_keypoints in dataset folder
    torch.save(common_keypoints, f'{calibrations_folder}/common_keypoints.pt')
    # also save xyzs_list items one by one
    for i in range(len(xyzs_list)):
        torch.save(xyzs_list[i], f'{calibrations_folder}/xyzs_list_{i}.pt')
        torch.save(rgbs_list[i], f'{calibrations_folder}/rgbs_list_{i}.pt')


    print('Starting optimization')

    num_of_views = common_keypoints.shape[1] // 3

    R = torch.nn.Parameter(torch.ones([num_of_views, 3]).to(device), requires_grad=True)
    T = torch.nn.Parameter(torch.zeros([num_of_views, 3]).to(device), requires_grad=True)

    # check if R, T exist

    if os.path.exists(f'{calibrations_folder}/R.pt') and os.path.exists(f'{calibrations_folder}/T.pt'):
        R = torch.load(f'{calibrations_folder}/R.pt').to(device)
        T = torch.load(f'{calibrations_folder}/T.pt').to(device)
        print('Found R, T, skipping optimization')
    else:
        opt = torch.optim.Adam([R, T], lr=1.0e-4)
        l1_loss = torch.nn.L1Loss()

        # for i in range(0, 10000):
        pbar = tqdm(range(0, 25000))
        for i in pbar:
            warped_list = []
            loss = 0
            for j in range(1, num_of_views):
                warped_list.append(common_keypoints[:, 3*j:3*j+3] @ axis_angle_to_matrix(R[j]) + T[j])
                loss += l1_loss(common_keypoints[:,0:3], warped_list[-1])
            # loss = l1_loss(common_keypoints[:,0:3], warped_1) +  l1_loss(common_keypoints[:,0:3], warped_2) + l1_loss(common_keypoints[:,0:3], warped_3)
            

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                # print(float(loss))
                pbar.set_description(f"Loss: {float(loss)}")


        torch.save(R, f'{calibrations_folder}/R.pt')
        torch.save(T, f'{calibrations_folder}/T.pt')


    warped_pcd_list = [torch.Tensor(xyzs_list[0]).to(device)]

    for i in range(1, num_of_views):
        warped_pcd_list.append(torch.Tensor(xyzs_list[i]).to(device) @ axis_angle_to_matrix(R[i]) + T[i])

    pcd_list = [item.detach().cpu().numpy() for item in warped_pcd_list]
    
    pcd = o3d.geometry.PointCloud()


    if(rgbs_list is None):
        colors = np.random.rand(num_of_views, 3).tolist()
        rgbs_list = []

        for pcd_item in pcd_list:
            rgbs_list.append(np.ones([pcd_item.shape[0], 3]) * colors.pop(0))

    pcd.points = o3d.utility.Vector3dVector(torch.cat([torch.tensor(pcd_item) for pcd_item in pcd_list], 0).detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(torch.cat([torch.tensor(rgb_item) for rgb_item in rgbs_list], 0).detach().cpu().numpy())
    
    o3d.io.write_point_cloud(f'{calibrations_folder}/calibration_pointcloud.ply', pcd)


    save_aabb_for_all_timeframes(args, 
                                num_views=num_of_views,
                                R=R,
                                T=T,
                                data_folder=calibrations_folder,
                                rgb_cams=rgb_cams,
                                depth_cams=depth_cams)


def save_aabb_for_all_timeframes(args, num_views, R, T, data_folder, rgb_cams, depth_cams):


    if(args.downscale_factor != 1):
        keys_to_downscale = ['f', 'c', 'camera_mtx']
        for i in range(len(rgb_cams)):
            for key in keys_to_downscale:

                if(key == 'camera_mtx'):
                    rgb_cams[i][key] = np.array(rgb_cams[i][key]).reshape([3,3])
                    depth_cams[i][key] = np.array(depth_cams[i][key]).reshape([3,3])

                    rgb_cams[i][key][:2, :] = rgb_cams[i][key][:2, :] / args.downscale_factor
                    depth_cams[i][key][:2, :] = depth_cams[i][key][:2, :] / args.downscale_factor

                rgb_cams[i][key] = [item / args.downscale_factor for item in rgb_cams[i][key]]
                depth_cams[i][key] = [item / args.downscale_factor for item in depth_cams[i][key]]

    files_to_consider = sorted(glob(data_folder + "/cam_0/rgb/*.png"))

    for i in range(len(files_to_consider)):
        files_to_consider[i] = files_to_consider[i].split("/")[-1]


    # now find aabb for each view
    # aabb_list = []
    aabb_best = None
    sparse_points_candidate = None
    sparse_points_candidate_colors = None

    for counter, file_to_consider in tqdm(enumerate(files_to_consider)):

        xyzs_list = []
        rgbs_list = []

        try:
            for i in range(num_views):
                # rgb_images_path = sorted(glob(data_folder + "/cam_*/rgb/*.png"))

                file_to_consider_formatted = file_to_consider.replace('video_0_frame_', f'video_{i}_frame_')
                rgb_image_path = glob(data_folder + f"/cam_{i}/rgb/{file_to_consider_formatted}")[0]
                depth_image_path = glob(data_folder + f"/cam_{i}/depth/{file_to_consider_formatted}")[0]

                # xyzs, rgbs, kpts_3d, kpts_2d = get_cloud(depth_images_path[i], rgb_images_path[i], i, depth_cams[i], rgb_cams[i])
                xyzs, rgbs, _, _ = get_cloud(depth_image_path, rgb_image_path, i, depth_cams[i], rgb_cams[i], do_aligning=False)

                if(i > 0):
                    xyzs = torch.tensor(xyzs,dtype=torch.float32).to('cuda') @ axis_angle_to_matrix(R[i]) + T[i]
                    xyzs = xyzs.detach().cpu().numpy()
                xyzs_list.append(xyzs)
                rgbs_list.append(rgbs)
        except:
            print(f'Could not find {file_to_consider} in all cameras, skipping')
            continue

        xyzs_list = np.concatenate(xyzs_list, 0)
        rgbs_list = np.concatenate(rgbs_list, 0)


        # visualize with open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs_list)
        pcd.colors = o3d.utility.Vector3dVector(rgbs_list)


        # Visualize original point cloud
        if(counter == 0):
            o3d.visualization.draw_geometries([pcd], "Original Point Cloud")

        # Apply statistical outlier removal
        nb_neighbors = 20  # Number of neighbors to consider for each point
        std_ratio = 0.3  # Standard deviation ratio

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # Select the inliers (non-outliers) for visualization
        inlier_cloud = pcd.select_by_index(ind)

        # Visualize the point cloud after outlier removal
        if(counter == 0):
            o3d.visualization.draw_geometries([inlier_cloud], "Point Cloud after Outlier Removal")

        inlier_colors = np.asarray(inlier_cloud.colors)
        inlier_cloud = np.asarray(inlier_cloud.points)
        # min_x, min_y, min_z = xyzs_list.min(axis=0)
        # max_x, max_y, max_z = xyzs_list.max(axis=0)  
        min_x, min_y, min_z = inlier_cloud.min(axis=0)
        max_x, max_y, max_z = inlier_cloud.max(axis=0)      

        # aabb_list.append([min_x, min_y, min_z, max_x, max_y, max_z])
        aabb_candidate = [min_x, min_y, min_z, max_x, max_y, max_z]
        print(f'Found aabb: {aabb_candidate}')

        if(aabb_best is None):
            aabb_best = aabb_candidate
            sparse_points_candidate = inlier_cloud
            sparse_points_candidate_colors = inlier_colors
        else:
            # check if inlier_cloud is outside of aabb_best, if so, update aabb_best to aabb_candidate
            if((inlier_cloud < aabb_best[0:3]).any() or (inlier_cloud > aabb_best[3:]).any()):
                # aabb_best = aabb_candidate
                # update aabb best to contain largest borders of both aabbs
                aabb_best[0:3] = np.minimum(aabb_best[0:3], aabb_candidate[0:3])
                aabb_best[3:] = np.maximum(aabb_best[3:], aabb_candidate[3:])

                sparse_points_candidate = inlier_cloud
                sparse_points_candidate_colors = inlier_colors
        print(f'Found aabb_best: {aabb_best}')
            

        # add aabb to the pointcloud

        pcd = o3d.geometry.PointCloud()

        aabb_points = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z]]
        ).reshape(-1, 3)

        pcd.points = o3d.utility.Vector3dVector(np.concatenate([inlier_cloud, aabb_points], 0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate([inlier_colors, np.ones([aabb_points.shape[0], 3]) * [1,0,0]], 0))

    
    o3d.visualization.draw_geometries([pcd], "Last Point Cloud after Outlier Removal and aabb shown")


    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(sparse_points_candidate)
    pcd.colors = o3d.utility.Vector3dVector(sparse_points_candidate_colors)

    o3d.io.write_point_cloud(f'{data_folder}/sparse_points_interest.ply', pcd)



    # aabb_tight = np.array(aabb_list).min(axis=0)
    aabb_tight = np.array(aabb_best) 


    # Separate the minimum and maximum values
    min_values = aabb_tight[:3]
    max_values = aabb_tight[3:]

    # Round down (floor) the minimum values to two decimal places
    rounded_min_values = np.sign(min_values) * np.floor(np.abs(min_values) * 100) / 100

    # Round up (ceiling) the maximum values to two decimal places
    rounded_max_values = np.sign(max_values) * np.ceil(np.abs(max_values) * 100) / 100


    rounded_aabb = np.concatenate((rounded_min_values, rounded_max_values))

    aabb_points = np.array([
        [rounded_min_values[0], rounded_min_values[1], rounded_min_values[2]],
        [rounded_min_values[0], rounded_min_values[1], rounded_max_values[2]],
        [rounded_min_values[0], rounded_max_values[1], rounded_min_values[2]],
        [rounded_min_values[0], rounded_max_values[1], rounded_max_values[2]],
        [rounded_max_values[0], rounded_min_values[1], rounded_min_values[2]],
        [rounded_max_values[0], rounded_min_values[1], rounded_max_values[2]],
        [rounded_max_values[0], rounded_max_values[1], rounded_min_values[2]],
        [rounded_max_values[0], rounded_max_values[1], rounded_max_values[2]]]
    ).reshape(-1, 3)

    pcd.points = o3d.utility.Vector3dVector(np.concatenate([sparse_points_candidate, aabb_points], 0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate([sparse_points_candidate_colors, np.ones([aabb_points.shape[0], 3]) * [1,0,0]], 0))

    o3d.visualization.draw_geometries([pcd], "Last Point Cloud after Outlier Removal and aabb shown once again (this was for PCA debugging)")
    
    # save aabb_tight in data_folder as list in a txt file
    with open(os.path.join(data_folder, 'aabb.txt'), 'w') as f:
        f.write(str(rounded_aabb.tolist()))

    print(f'Found aabb_tight: {aabb_tight}')
    print(f'Saved aabb_tight in {data_folder}/aabb.txt')


        






def read_video_mkv(filename,
                    #  begin_frame,
                        # end_frame,
                        # by_step,
                        out_folder,
                        file_pattern="video_4_frame_",
                        downscale_factor=1,
                        last_frame=-1):
    

    with PyK4APlayback(filename) as playback:

        # Get the calibration data
        calibration_data = playback.calibration

        # print(calibration_data)
        rgb_camera_matrix = calibration_data.get_camera_matrix(calibration.CalibrationType.COLOR)
        depth_camera_matrix = calibration_data.get_camera_matrix(calibration.CalibrationType.DEPTH)

        ext_color2depth = calibration_data.get_extrinsic_parameters(calibration.CalibrationType.COLOR, calibration.CalibrationType.DEPTH)
        rotation, translation = ext_color2depth[0], ext_color2depth[1]
        ext_color2depth = np.concatenate((rotation, translation.reshape(3,1)), axis=1)
        ext_color2depth = np.concatenate((ext_color2depth, np.array([[0,0,0,1.0]])), axis=0)

        ext_depth2color = calibration_data.get_extrinsic_parameters(calibration.CalibrationType.DEPTH, calibration.CalibrationType.COLOR)
        rotation, translation = ext_depth2color[0], ext_depth2color[1]
        ext_depth2color = np.concatenate((rotation, translation.reshape(3,1)), axis=1)
        # add 0,0,0,1 to the last row
        ext_depth2color = np.concatenate((ext_depth2color, np.array([[0,0,0,1.0]])), axis=0)

        k_color = calibration_data.get_distortion_coefficients(calibration.CalibrationType.COLOR)
        k_depth = calibration_data.get_distortion_coefficients(calibration.CalibrationType.DEPTH)

        rgb_cam_candidate = {
            'c' : rgb_camera_matrix[0:2, 2].tolist(),
            # 'f' : rgb_camera_matrix[0:2, 0:2],
            'f' : [rgb_camera_matrix[0, 0], rgb_camera_matrix[1, 1]],
            'camera_mtx': rgb_camera_matrix.tolist(),
            'ext_color2depth': ext_color2depth.tolist(),
            'k': k_color.tolist()
        }

        depth_cam_candidate = {
            'c' : depth_camera_matrix[0:2, 2].tolist(),
            # 'f' : depth_camera_matrix[0:2, 0:2],
            'f' : [depth_camera_matrix[0, 0], depth_camera_matrix[1, 1]],
            'camera_mtx': depth_camera_matrix.tolist(),
            'ext_depth2color': ext_depth2color.tolist(),
            'k': k_depth.tolist()
        }



    if not os.path.exists(out_folder):
        print('Creating folder: ', out_folder)
        os.makedirs(out_folder)
        os.makedirs(os.path.join(out_folder, "depth"))
        os.makedirs(os.path.join(out_folder, "rgb"))
        os.makedirs(os.path.join(out_folder, "mask"))


    # save the calibration data in separate files
    with open(os.path.join(out_folder, 'rgb_camera.json'), 'w') as fp:
        json.dump(rgb_cam_candidate, fp, indent=4)
    
    with open(os.path.join(out_folder, 'depth_camera.json'), 'w') as fp:
        json.dump(depth_cam_candidate, fp, indent=4)

    splits = filename.split("/")
    directory = "/".join(splits[:-1])
    video_file_name = splits[-1].split(".")[0]

    mask_directory = f'{directory}/masks/{video_file_name}'
    masks = sorted(glob(f'{mask_directory}/*.png'))


    # ffmpeg -i {filename} -map 0:0 {out_folder}/rgb/{file_pattern}_%04d.png
    do_system(f"ffmpeg -i {filename} -map 0:0 {out_folder}/rgb/{file_pattern}%04d.png")

    # ffmpeg -i {filename} -map 0:1 {out_folder}/depth/{file_pattern}_%04d.png
    do_system(f"ffmpeg -i {filename} -map 0:1 {out_folder}/depth/{file_pattern}%04d.png")

    rgb_images = sorted(glob(f'{out_folder}/rgb/*.png'))
    depth_images = sorted(glob(f'{out_folder}/depth/*.png'))

    # Seek and read frames from begin_frame to end_frame with by_step step

    num_frames = len(masks) if last_frame == -1 else last_frame

    if('charuco' in out_folder):
        num_frames = 1



    for ith_frame in tqdm(range(num_frames)):

        # read rgb image

        try:
            rgb = cv2.imread(rgb_images[ith_frame])
            # read depth image
            depth = cv2.imread(depth_images[ith_frame], flags=-1)
        except IndexError:
            print('IndexError, finishing on frame: ', ith_frame)
            ith_frame = ith_frame - 1
            break
        # read mask image
        # mask = cv2.imread(masks[ith_frame], flags=-1)

        # combine mask and color image
        if('charuco' in filename):
            # image_shape = rgbd.color.shape
            image_shape = rgb.shape
            mask = np.ones((image_shape[0], image_shape[1])).astype(np.bool_)          
        else:
            mask = cv2.imread(masks[ith_frame], flags=-1)

            if(len(mask.shape) == 3 and mask.shape[2] == 3):
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0
            elif(len(mask.shape) == 3 and mask.shape[2] == 4):
                mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY) > 0
            else:
                mask = mask > 0

        color_image = np.array(rgb)
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image[mask == 0] = 0

        # remove green spill from edges
        # if('charuco' not in filename):
            # color_image = remove_green_spill_from_edges(color_image, mask)
        if('charuco' not in filename):
            color_image = remove_green_spill_from_edges(color_image, mask)
            color_image, mask = smooth_edges(color_image, mask)

        # save color image
        cv2.imwrite(rgb_images[ith_frame], color_image)
        # save mask image
        cv2.imwrite(f'{out_folder}/mask/{file_pattern}{str(ith_frame+1).zfill(4)}.png', mask.astype(np.uint8)*255)
        # save depth image

        if(args.align_depth):
            # TODO check if you want aligned depth image immediately or later
            _, depth, _ = get_aligned_rgbd(
                depth_images[ith_frame],
                rgb_images[ith_frame],
                depth_cam_candidate,
                rgb_cam_candidate
            )
            
            
            depth[mask==0] = 0


            # print(f'first extracted depth image max {capture_data.depth.max()}, aligned_depth image max {depth.max()}')
            if(downscale_factor != 1):
                depth = cv2.resize(depth, (0,0), fx=1/downscale_factor, fy=1/downscale_factor)

            cv2.imwrite(depth_images[ith_frame], (depth * 1000.0).astype(np.uint16))
        else:
            depth = transformation.depth_image_to_color_camera(depth, calibration=calibration_data, thread_safe=True)
            depth[mask==0] = 0
            cv2.imwrite(depth_images[ith_frame], depth.astype(np.uint16))

    return ith_frame+1


def get_charuco_transform(keypoints):

    K = keypoints
    center = torch.mean(K, axis=0)
    K_centered = K - center
    _, _, R = torch.linalg.svd(K_centered)

    return center, R

def generate_grid_basis(grid_size=32, n_dims=3, minv=[-1.0, -1.0, -1.0], maxv=[1.0, 1.0, 1.0]):
    """ Generate d-dimensional grid BPS basis
    Parameters
    ----------
    grid_size: int
        number of elements in each grid axe
    minv: float
        minimum element of the grid
    maxv
        maximum element of the grid
    Returns
    -------
    basis: numpy array [grid_size**n_dims, n_dims]
        n-d grid points
    """

    linspaces = [np.linspace(minv[d], maxv[d], num=grid_size) for d in range(0, n_dims)]
    coords = np.meshgrid(*linspaces)
    basis = np.concatenate([coords[i].reshape([-1, 1]) for i in range(0, n_dims)], axis=1)

    return basis
    
def run_transform_charuco_calibration2neus(args):
    # begin_frame = args.begin_frame
    # end_frame = args.end_frame
    # by_step = args.by_step
    # args.neus_frames = len(range(begin_frame, end_frame, by_step))

    neus_frames = 100000
    num_cams = len(glob(args.src_dir + f"/cam_*"))
    for i in range(num_cams):
        neus_frames = min(neus_frames, len(glob(args.src_dir + f"/cam_{i}/rgb/*.png")))
    args.neus_frames = neus_frames
    # assert args.neus_frames == len(glob(args.src_dir + f"/cam_0/rgb/*.png")), "Number of frames in the folder doesn't match the number of frames in the video"

    # Load R.pt and T.pt

    R = torch.load(f'{args.src_dir}/R.pt')
    T = torch.load(f'{args.src_dir}/T.pt')


    num_of_views = R.shape[0]
    # Load rgb_camera.json and depth_camera.json
    rgb_cams = []
    depth_cams = []
    for i in range(num_of_views):
        with open(f'{args.src_dir}/cam_{i}/rgb_camera.json') as f:
            rgb_cam = json.load(f)
        with open(f'{args.src_dir}/cam_{i}/depth_camera.json') as f:
            depth_cam = json.load(f)

        rgb_cams.append(rgb_cam)
        depth_cams.append(depth_cam)

    if(args.downscale_factor != 1):
        keys_to_downscale = ['f', 'c', 'camera_mtx']
        for i in range(len(rgb_cams)):
            for key in keys_to_downscale:

                if(key == 'camera_mtx'):
                    rgb_cams[i][key] = np.array(rgb_cams[i][key]).reshape([3,3])
                    depth_cams[i][key] = np.array(depth_cams[i][key]).reshape([3,3])

                    rgb_cams[i][key][:2, :] = rgb_cams[i][key][:2, :] / args.downscale_factor
                    depth_cams[i][key][:2, :] = depth_cams[i][key][:2, :] / args.downscale_factor

                rgb_cams[i][key] = [item / args.downscale_factor for item in rgb_cams[i][key]]
                depth_cams[i][key] = [item / args.downscale_factor for item in depth_cams[i][key]]



    cameras = get_torch_cameras(rgb_cams, R, T)

    
    n_frames = args.neus_frames

    cam_dict = dict()
    # n_views = len(poses_raw)
    n_views = num_of_views

    cam_dict = {f'view_{i}' : {} for i in range(n_views)}

    sparse_interest_pcd = o3d.io.read_point_cloud(os.path.join(args.src_dir, 'sparse_points_interest.ply'))

    
    first_camera = cameras[0]
    R_first = first_camera.R[0].detach().cpu().numpy()
    T_first = first_camera.T[0].detach().cpu().numpy()

    rgb_shape = [1080, 1920, 3]
    R_first, T_first, K = opencv_from_cameras_projection(cameras[0], torch.as_tensor(rgb_shape)[None,:2])
    R_first = R_first[0].detach().cpu().numpy()
    T_first = T_first[0].detach().cpu().numpy()


    for frame_num in tqdm(range(n_frames)):
        for view_num in range(n_views):
            intrinsic = np.eye(4).astype(np.float32)

            intrinsic[:3,:3] = np.array(depth_cams[view_num]['camera_mtx'])

            # TODO old_R and old_T should be removed 
            old_R = cameras[view_num].R[0].detach().cpu().numpy()
            old_T = cameras[view_num].T[0].detach().cpu().numpy()

            chosen_camera = cameras[view_num]

            chosen_camera.R = chosen_camera.R.to(chosen_camera.device)
            chosen_camera.T = chosen_camera.T.to(chosen_camera.device)

            R, T, K = opencv_from_cameras_projection(chosen_camera, torch.as_tensor(rgb_shape).to(chosen_camera.device)[None,:2])
            R = R[0].detach().cpu().numpy()
            T = T[0].detach().cpu().numpy()
            K_init = K[0]
            intrinsic = np.eye(4)
            intrinsic[:3, :3] = K_init.detach().cpu().numpy()

            pose = np.concatenate([R, T.reshape([3, 1])], 1)
            pose = np.concatenate([pose, np.array([[0,0,0,1.0]])], 0)

            pose = np.linalg.inv(pose) # w2c->c2w
            w2c = np.linalg.inv(pose) # c2w->w2c
            world_mat = intrinsic @ w2c
            # cam_dict['camera_mat_{}'.format(i)] = intrinsic
            cam_dict[f'view_{view_num}']['camera_mat_{}'.format(frame_num)] = intrinsic
            # cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
            cam_dict[f'view_{view_num}']['camera_mat_inv_{}'.format(frame_num)] = np.linalg.inv(intrinsic)
            # cam_dict['world_mat_{}'.format(i)] = world_mat
            cam_dict[f'view_{view_num}']['world_mat_{}'.format(frame_num)] = world_mat
            # cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)
            cam_dict[f'view_{view_num}']['world_mat_inv_{}'.format(frame_num)] = np.linalg.inv(world_mat)


        pcd = sparse_interest_pcd



        vertices = pcd.points
        bbox_max = np.max(vertices, axis=0)
        bbox_min = np.min(vertices, axis=0)
        center = (bbox_max + bbox_min) * 0.5
        radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
        scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
        scale_mat[:3, 3] = center
        # scale_mat = np.eye(4).astype(np.float32)

        



        for view_num in range(n_views):
            cam_dict[f'view_{view_num}']['scale_mat_{}'.format(frame_num)] = scale_mat
            cam_dict[f'view_{view_num}']['scale_mat_inv_{}'.format(frame_num)] = np.linalg.inv(scale_mat)


    pcd = sparse_interest_pcd

 

    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points, dtype=np.float32) @ scale_mat[:3,:3].T - scale_mat[:3,3])

    # convert bbox with scale mat

    use_pca = False
    if(use_pca == True):
        # use PCA to find the best orientation of aabb_tight
        from sklearn.decomposition import PCA
        sparse_points_candidate = np.asarray(pcd.points)
        centroid = np.mean(sparse_points_candidate, axis=0)
        sparse_points_candidate = sparse_points_candidate - centroid

        pca = PCA(n_components=3)
        pca.fit(sparse_points_candidate)

        point_cloud_pca = pca.transform(sparse_points_candidate)

        # find the min and max values of the point cloud
        min_values = point_cloud_pca.min(axis=0)
        max_values = point_cloud_pca.max(axis=0)

        # return the min and max values to the original coordinate system
        min_values = pca.inverse_transform(min_values) + centroid
        max_values = pca.inverse_transform(max_values) + centroid

        bbox_min = min_values
        bbox_max = max_values

        # Check if the orientation of the min and max values is correct
        # If not, swap the values
        for i in range(len(min_values)):
            if min_values[i] > max_values[i]:
                min_values[i], max_values[i] = max_values[i], min_values[i]

    else:
        vertices = pcd.points
        bbox_max = np.max(vertices, axis=0)
        bbox_min = np.min(vertices, axis=0)

    # round to 2 decimals
    rounded_min_values = np.sign(bbox_min) * np.floor(np.abs(bbox_min) * 100) / 100
    rounded_max_values = np.sign(bbox_max) * np.ceil(np.abs(bbox_max) * 100) / 100


    # bbox_min = np.round(bbox_min, 2)
    # bbox_max = np.round(bbox_max, 2)

    # aabb = [*(1*bbox_min), *(1*bbox_max)]
    aabb = [*(1.1*rounded_min_values), *(1.1*rounded_max_values)]

    with open(os.path.join(args.src_dir, 'aabb.txt'), 'w') as f:
        # write it as a list
        f.write(str(aabb))

    print('Found aabb after scale_mat, ', aabb)

    # Visualize the point cloud after scale_mat and aabb

    aabb_points = np.array([
        [aabb[0], aabb[1], aabb[2]],
        [aabb[0], aabb[1], aabb[5]],
        [aabb[0], aabb[4], aabb[2]],
        [aabb[0], aabb[4], aabb[5]],
        [aabb[3], aabb[1], aabb[2]],
        [aabb[3], aabb[1], aabb[5]],
        [aabb[3], aabb[4], aabb[2]],
        [aabb[3], aabb[4], aabb[5]]]
    ).reshape(-1, 3)

    pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.points), aabb_points], 0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.colors), np.ones([aabb_points.shape[0], 3]) * [1,0,0]], 0))
    o3d.visualization.draw_geometries([pcd], "(last last) Last Point Cloud after scale_mat and aabb shown")


    custom_aabb = np.array([-0.43, -0.65, -0.65, 0.42, 0.6, 0.41])

    aabb_points = np.array([
        [custom_aabb[0], custom_aabb[1], custom_aabb[2]],
        [custom_aabb[0], custom_aabb[1], custom_aabb[5]],
        [custom_aabb[0], custom_aabb[4], custom_aabb[2]],
        [custom_aabb[0], custom_aabb[4], custom_aabb[5]],
        [custom_aabb[3], custom_aabb[1], custom_aabb[2]],
        [custom_aabb[3], custom_aabb[1], custom_aabb[5]],
        [custom_aabb[3], custom_aabb[4], custom_aabb[2]],
        [custom_aabb[3], custom_aabb[4], custom_aabb[5]]]
    ).reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.points), aabb_points], 0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.colors), np.ones([aabb_points.shape[0], 3]) * [0,1,0]], 0))
    o3d.visualization.draw_geometries([pcd], "(last last) Last Point Cloud that works with custom_aabb")

    for view_num in range(n_views):
        np.savez(os.path.join(args.src_dir, f'cam_{view_num}/cameras_sphere.npz'), **cam_dict[f'view_{view_num}'])
    
    
    print('Process done!')




if __name__ == '__main__':

   

    parser = argparse.ArgumentParser(description="Run COLMAP with new images and optional image list.")
    args = parse_args()


    # DATA_ROOT = "ginormous_dataset"
    # DATA_ROOT = "plane_dataset"
    # DATA_ROOT= "../../../Downloads/green_screen_3/plane/"
    DATA_ROOT="../../../Downloads/green_screen_5/activities/book/"
    static_video_paths = [
        f'{DATA_ROOT}/cam_0.mkv',
        f'{DATA_ROOT}/cam_1.mkv',
        f'{DATA_ROOT}/cam_2.mkv',
        f'{DATA_ROOT}/cam_3.mkv',
        f'{DATA_ROOT}/cam_4.mkv'
    ]    
    dynamic_video_paths = [
        # f'{DATA_ROOT}/3_video_with_charuco.mkv'
    ]

    video_config = {
        "static_videos": [f"{i}" for i in range(len(static_video_paths))],  # All videos except the last one are static for this example
        "dynamic_videos": [f"{i + len(static_video_paths)}" for i in range(len(dynamic_video_paths))], # Only the last video is dynamic for this example
    }

    args.video_config = video_config

    # TODO: add vieo extractor from which to extract frames from make_dataset.py
    if args.run_make_dataset:
        # For NeuS compatible format it goes a bit different and more convenient for later
        min_num_frames = 10000
        last_frame = -1
        for video_idx in video_config["static_videos"]:
            last_frame = read_video_mkv(static_video_paths[int(video_idx)], 
                        # args.begin_frame, 
                        # args.end_frame, 
                        # args.by_step, 
                        args.src_dir + f"/cam_{video_idx}",
                        file_pattern=f"video_{video_idx}_frame_",
                        downscale_factor=args.downscale_factor,
                        last_frame=last_frame)
            min_num_frames = min(min_num_frames, last_frame)
        for video_idx in video_config["dynamic_videos"]:
            last_frame = read_video_mkv(dynamic_video_paths[int(video_idx) - len(static_video_paths)], 
                        # args.begin_frame, 
                        # args.end_frame, 
                        # args.by_step, 
                        args.src_dir + f"/cam_{video_idx}",
                        file_pattern=f"video_{video_idx}_frame_",
                        downscale_factor=args.downscale_factor,
                        last_frame=last_frame)
            min_num_frames = min(min_num_frames, last_frame)


        # go through all the folder with masks,depths/rgbs and delte last frames

        cam_folders = glob(f'{args.src_dir}/cam_*')

        for cam_folder in cam_folders:
            rgb_folder = f'{cam_folder}/rgb'
            depth_folder = f'{cam_folder}/depth'
            mask_folder = f'{cam_folder}/mask'

            rgb_images = sorted(glob(f'{rgb_folder}/*.png'))
            depth_images = sorted(glob(f'{depth_folder}/*.png'))
            mask_images = sorted(glob(f'{mask_folder}/*.png'))

            for rgb_image in rgb_images[min_num_frames:]:
                os.remove(rgb_image)
            for depth_image in depth_images[min_num_frames:]:
                os.remove(depth_image)
            for mask_image in mask_images[min_num_frames:]:
                os.remove(mask_image)
                

        args.neus_frames = min_num_frames


            

    # add running first step
    if args.run_calibration_finding:
        choice = input("Do you need to run initial calibration step (i.e. with charuco)? (yes/no)").strip().lower()

        backup_images = args.src_dir
        # backup_begin_frame = args.begin_frame
        
        if(choice == 'yes'):
            # TODO Dusan: bring back prompts
            choice = input("Enter images folder: ").strip().lower()

            args.src_dir = choice
            # args.begin_frame = 1
            # args.begin_frame 
            images_list = glob(f'{args.src_dir}/*.png')
            try:
                # TODO Dusan: unify this
                args.begin_frame = int(images_list[0].split('/')[-1].split('frame_')[1].split('.')[0])
            except:
                images_list = glob(f'{args.src_dir}/cam_*/rgb/*.png')
                args.begin_frame = int(images_list[0].split('/')[-1].split('frame_')[1].split('.')[0])


        if(args.run_calibration_finding):
            run_calibration_finding(args, calibrations_folder=backup_images)

        args.src_dir = backup_images
        # args.begin_frame = backup_begin_frame



    if args.run_transform_charuco_calibration2neus:
        run_transform_charuco_calibration2neus(args)
