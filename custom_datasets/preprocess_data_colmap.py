
import argparse
from glob import glob
import os
import os
from tqdm import tqdm
import open3d as o3d
import numpy as np
from pyk4a import PyK4APlayback, ImageFormat, calibration
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



def parse_args():
    parser = argparse.ArgumentParser(description="Use charuco calibration for dataset creation.")

    parser.add_argument("--run_calibration_finding", action="store_true", help="Running calibriation for overlapping pointclouds")
    parser.add_argument("--run_transform_charuco_calibration2neus", action="store_true", help="Converting calibration format to neus format")

    parser.add_argument("--run_make_dataset", action="store_true", help="Run make_dataset.py on the new images.")
    parser.add_argument("--begin_frame", type=int, default=0, help="Frame number to begin reading from.")
    parser.add_argument("--end_frame", type=int, default=-1, help="Frame number to stop reading at.")
    parser.add_argument("--by_step", type=int, default=33333, help="Step size between frames to read.")
    parser.add_argument("--src_dst", type=str, help="Folder path to save the extracted images.")

    args = parser.parse_args()
    return args



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

    depth_image_reprojected = np.zeros([1080, 1920])
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

def remove_green_spill_from_edges(img, mask, threshold=0.3, spill_factor=0.5):
    # Identify edges of the mask
    sobel_x = ndimage.sobel(mask, axis=0, mode='constant')
    sobel_y = ndimage.sobel(mask, axis=1, mode='constant')
    edge_magnitude = np.hypot(sobel_x, sobel_y)
    edges = edge_magnitude > threshold

    # Reduce the green component on the edges

    # edges = edges[:,:,0]
    img[edges, 1] = (img[edges, 1] * spill_factor).astype(np.uint8)
    return img

# def get_cloud(depth_image_path, rgb_image_path, cam_id, depth_cam, rgb_cam, mask=None):
def get_cloud(depth_image_path, rgb_image_path, cam_id, depth_cam, rgb_cam, mask=None, do_aligning=True):

    
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

    kpts, ids = detect_aruco_markers(rgb_image_path,visualize=False)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(uvd[:, 0:2])
    kpts_3d = []
    for ix in range(0, len(kpts)):
        distances, indices = nbrs.kneighbors(kpts[ix][0])
        kpts_3d.append(xyz[indices].reshape([-1, 3]))

    return xyz, rgb, [kpts_3d, ids], [kpts, ids]

def detect_aruco_markers(img_path, visualize=True):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()

    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

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
  #Ts = []
  #Rs = []
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
    # rgb_images_path = os.listdir(args.src_dst)
    rgb_images_path = sorted(glob(args.src_dst + "/cam_*/rgb/*.png"))
    depth_images_path = sorted(glob(args.src_dst + "/cam_*/depth/*.png"))

    depth_cams = []
    rgb_cams = []

    for i in range(len(rgb_images_path)):
        with open(f'{calibrations_folder}/cam_{i}/rgb_camera.json') as f:
            rgb_cams.append(json.load(f))
        with open(f'{calibrations_folder}/cam_{i}/depth_camera.json') as f:
            depth_cams.append(json.load(f))


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
            xyzs, rgbs, kpts_3d, kpts_2d = get_cloud(depth_images_path[i], rgb_images_path[i], i, depth_cams[i], rgb_cams[i])
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
        print('Found T, skipping optimization')
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

    # colors = [
    #     [1, 0, 0],  # Red
    #     [0, 1, 0],  # Green
    #     [0, 0, 1],  # Blue
    #     [1, 1, 0],  # Yellow
    #     [1, 0, 1],  # Magenta
    #     [1
    # ]
    # make random colors
    

    # if(len(pcd_list) > len(colors)):
        # raise Exception('Not enough colors for pointclouds')
    
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
        std_ratio = 0.5  # Standard deviation ratio

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

        if(counter == 0):
            o3d.visualization.draw_geometries([pcd], "Point Cloud after Outlier Removal and aabb shown")

        # inlier_cloud.points = o3d.utility.Vector3dVector(xyzs_list)
        # inlier_cloud.colors = o3d.utility.Vector3dVector(rgbs_list)
        # add aabb 


    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(sparse_points_candidate)
    pcd.colors = o3d.utility.Vector3dVector(sparse_points_candidate_colors)

    o3d.io.write_point_cloud(f'{data_folder}/sparse_points_interest.ply', pcd)



    # aabb_tight = np.array(aabb_list).min(axis=0)
    aabb_tight = aabb_best

    # save aabb_tight in data_folder as list in a txt file
    np.savetxt(f'{data_folder}/aabb.txt', aabb_tight) 

    print(f'Found aabb_tight: {aabb_tight}')
    print(f'Saved aabb_tight in {data_folder}/aabb.txt')


        







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
        os.makedirs(os.path.join(out_folder, "mask"))


    def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
        # examples for all possible pyk4a.ColorFormats
        if color_format == ImageFormat.COLOR_MJPG:
            color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        elif color_format == ImageFormat.COLOR_NV12:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        elif color_format == ImageFormat.COLOR_YUY2:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
        return color_image



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


    # save the calibration data in separate files
    with open(os.path.join(out_folder, 'rgb_camera.json'), 'w') as fp:
        json.dump(rgb_cam_candidate, fp, indent=4)
    
    with open(os.path.join(out_folder, 'depth_camera.json'), 'w') as fp:
        json.dump(depth_cam_candidate, fp, indent=4)

    calibration_metadata = reader.get_metadata()

     
    if(calibration_metadata.stream_length_usec < end_frame):
        end_frame = calibration_metadata.stream_length_usec
        print(f"End frame is too big, setting it to {end_frame}")
    

    reader.close()

    splits = filename.split("/")
    directory = "/".join(splits[:-1])
    video_file_name = splits[-1].split(".")[0]

    mask_directory = f'{directory}/masks/{video_file_name}'

    masks = sorted(glob(f'{mask_directory}/*.png'))
    # for mask in masks:
    #     mask_filename = os.path.join(out_folder, f"masks/{mask.split('/')[-1]}")

    with PyK4APlayback(filename) as playback:
        for mask_num, ith_frame in tqdm(enumerate(range(begin_frame, end_frame, by_step))):
            playback.seek(int(ith_frame))
            capture_data = playback.get_next_capture()
            # if(capture_data.color is None):
            counter = 1
            while(capture_data.color is None and int(ith_frame) + counter != end_frame):
                counter += 1
                playback.seek(int(ith_frame) + counter)
                capture_data = playback.get_next_capture()
                
            if(capture_data.color is None or int(ith_frame) + counter > end_frame):
                print(f'Frame {ith_frame} is None, skipping')
                continue
            # i = 1

            color_filename = os.path.join(out_folder, f"rgb/{file_pattern}{ith_frame}.png")
            depth_filename = os.path.join(out_folder, f"depth/{file_pattern}{ith_frame}.png")
            mask_filename = os.path.join(out_folder, f"mask/{file_pattern}{ith_frame}.png")

            # combine mask and color image
            mask = cv2.imread(masks[mask_num], flags=-1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 180

            color_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture_data.color)
            

            color_image[mask == 0] = 0 

            color_image = remove_green_spill_from_edges(color_image, mask)
            cv2.imwrite(color_filename, color_image)        
            cv2.imwrite(depth_filename, capture_data.depth.astype(np.uint16))

            if(True):
                # TODO check if you want aligned depth image immediately or later
                depth_image = capture_data.depth
                _, depth_image, _ = get_aligned_rgbd(
                    depth_filename,
                    color_filename,
                    depth_cam_candidate,
                    rgb_cam_candidate
                )
                
                
                depth_image[mask==0] = 0


                print(f'first extracted depth image max {capture_data.depth.max()}, aligned_depth image max {depth_image.max()}')
                cv2.imwrite(depth_filename, (depth_image * 1000.0).astype(np.uint16))

            cv2.imwrite(mask_filename, mask.astype(np.uint8) * 255)

    # Now run the same for masks


def run_transform_charuco_calibration2neus(args):
    begin_frame = args.begin_frame
    end_frame = args.end_frame
    by_step = args.by_step
    args.neus_frames = len(range(begin_frame, end_frame, by_step))

    assert args.neus_frames == len(glob(args.src_dst + f"/cam_0/rgb/*.png")), "Number of frames in the folder doesn't match the number of frames in the video"

    # Load R.pt and T.pt

    R = torch.load(f'{args.src_dst}/R.pt')
    T = torch.load(f'{args.src_dst}/T.pt')


    num_of_views = R.shape[0]
    # Load rgb_camera.json and depth_camera.json
    rgb_cams = []
    depth_cams = []
    for i in range(num_of_views):
        with open(f'{args.src_dst}/cam_{i}/rgb_camera.json') as f:
            rgb_cam = json.load(f)
        with open(f'{args.src_dst}/cam_{i}/depth_camera.json') as f:
            depth_cam = json.load(f)

        rgb_cams.append(rgb_cam)
        depth_cams.append(depth_cam)


    cameras = get_torch_cameras(rgb_cams, R, T)

    
    n_frames = args.neus_frames

    cam_dict = dict()
    # n_views = len(poses_raw)
    n_views = num_of_views

    cam_dict = {f'view_{i}' : {} for i in range(n_views)}

    sparse_interest_pcd = o3d.io.read_point_cloud(os.path.join(args.src_dst, 'sparse_points_interest.ply'))

    
    first_camera = cameras[0]
    R_first = first_camera.R[0].detach().cpu().numpy()
    T_first = first_camera.T[0].detach().cpu().numpy()

    rgb_shape = [1080, 1920, 3]
    R_first, T_first, K = opencv_from_cameras_projection(cameras[0], torch.as_tensor(rgb_shape)[None,:2])
    R_first = R_first[0].detach().cpu().numpy()
    T_first = T_first[0].detach().cpu().numpy()


    for frame_num in range(n_frames):
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


        # pcd = trimesh.load(os.path.join(args.src_dst, 'sparse_points_interest.ply'))
        # pcd = o3d.io.read_point_cloud(os.path.join(args.src_dst, 'sparse_points_interest.ply'))
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

    vertices = pcd.points
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)

    aabb = [*(1*bbox_min), *(1*bbox_max)]

    with open(os.path.join(args.src_dst, 'aabb.txt'), 'w') as f:
        # write it as a list
        f.write(str(aabb))

    # np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    # save aabb to a txt file

    # aabb = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    # np.savetxt(os.path.join(args.src_dst, 'aabb.txt'), aabb)
    for view_num in range(n_views):
        np.savez(os.path.join(args.src_dst, f'cam_{view_num}/cameras_sphere.npz'), **cam_dict[f'view_{view_num}'])
    
    
    print('Process done!')




if __name__ == '__main__':

   

    parser = argparse.ArgumentParser(description="Run COLMAP with new images and optional image list.")
    args = parse_args()


    # DATA_ROOT = "ginormous_dataset"
    # DATA_ROOT = "plane_dataset"
    DATA_ROOT= "../../../Downloads/green_screen_3/plane/"
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

    # TODO: add video extractor from which to extract frames from make_dataset.py
    if args.run_make_dataset:
        # For NeuS compatible format it goes a bit different and more convenient for later
        for video_idx in video_config["static_videos"]:
            read_video_mkv(static_video_paths[int(video_idx)], 
                        args.begin_frame, 
                        args.end_frame, 
                        args.by_step, 
                        args.src_dst + f"/cam_{video_idx}",
                        file_pattern=f"video_{video_idx}_frame_")
        for video_idx in video_config["dynamic_videos"]:
            read_video_mkv(dynamic_video_paths[int(video_idx) - len(static_video_paths)], 
                        args.begin_frame, 
                        args.end_frame, 
                        args.by_step, 
                        args.src_dst + f"/cam_{video_idx}",
                        file_pattern=f"video_{video_idx}_frame_")
            

            

    # add running first step
    if args.run_calibration_finding:
        choice = input("Do you need to run initial calibration step (i.e. with charuco)? (yes/no)").strip().lower()

        backup_images = args.src_dst
        backup_begin_frame = args.begin_frame
        
        if(choice == 'yes'):
            # TODO Dusan: bring back prompts
            choice = input("Enter images folder: ").strip().lower()

            args.src_dst = choice
            # args.begin_frame = 1
            # args.begin_frame 
            images_list = glob(f'{args.src_dst}/*.png')
            try:
                # TODO Dusan: unify this
                args.begin_frame = int(images_list[0].split('/')[-1].split('frame_')[1].split('.')[0])
            except:
                images_list = glob(f'{args.src_dst}/cam_*/rgb/*.png')
                args.begin_frame = int(images_list[0].split('/')[-1].split('frame_')[1].split('.')[0])


        if(args.run_calibration_finding):
            run_calibration_finding(args, calibrations_folder=backup_images)

        args.src_dst = backup_images
        args.begin_frame = backup_begin_frame



    if args.run_transform_charuco_calibration2neus:
        run_transform_charuco_calibration2neus(args)
