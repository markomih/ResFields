import numpy as np
import os
import sys
import imageio
import skimage.transform
import trimesh
import cv2 as cv
from glob import glob
import colmap_read_model as read_model

def load_colmap_data(realdir):
    """
    Load the colmap data from the sparse folder inside the realdir
    """
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    # camerasfile = os.path.join(realdir, '0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    # imagesfile = os.path.join(realdir, '0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    # points3dfile = os.path.join(realdir, '0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm

def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts = np.stack(pts_arr, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(basedir, 'sparse_points.ply'))

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape )

    poses = np.moveaxis(poses, -1, 0)
    poses = poses[perm]
    np.save(os.path.join(basedir, 'poses.npy'), poses)



def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')     
            
def gen_poses(basedir, match_type, factors=None):
    
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        raise NotImplementedError
        # run_colmap(basedir, match_type)
    else:
        print('Don\'t need to run COLMAP')
        
    print('Post-colmap')
    
    poses, pts3d, perm = load_colmap_data(basedir)


    save_poses(basedir, poses, pts3d, perm)
    
    if factors is not None:
        print( 'Factors:', factors)
        minify(basedir, factors)
    
    print( 'Done with imgs2poses' )
    
    return True


def gen_cameras_dynamic(args):
    # work_dir = sys.argv[1]
    work_dir = f'{args.input_model_path}_neus'
    # n_frames = int(sys.argv[2])
    n_frames = args.neus_frames

    poses_hwf = np.load(os.path.join(work_dir, 'poses.npy')) # n_views, 3, 5
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]
    pts = []
    pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
    pts = np.stack(pts, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(work_dir, 'pose.ply'))
    #

    cam_dict = dict()
    n_views = len(poses_raw)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    cam_dict = {f'view_{i}' : {} for i in range(n_views)}

    for frame_num in range(n_frames):
        for view_num in range(n_views):
            pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            pose[:3, :4] = poses_raw[view_num]
            pose = pose @ convert_mat
            # h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
            h, w, f = hwf[view_num, 0], hwf[view_num, 1], hwf[view_num, 2]
            intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
            intrinsic[0, 2] = (w - 1) * 0.5
            intrinsic[1, 2] = (h - 1) * 0.5
            w2c = np.linalg.inv(pose)
            world_mat = intrinsic @ w2c
            world_mat = world_mat.astype(np.float32)
            # cam_dict['camera_mat_{}'.format(i)] = intrinsic
            cam_dict[f'view_{view_num}']['camera_mat_{}'.format(frame_num)] = intrinsic
            # cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
            cam_dict[f'view_{view_num}']['camera_mat_inv_{}'.format(frame_num)] = np.linalg.inv(intrinsic)
            # cam_dict['world_mat_{}'.format(i)] = world_mat
            cam_dict[f'view_{view_num}']['world_mat_{}'.format(frame_num)] = world_mat
            # cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)
            cam_dict[f'view_{view_num}']['world_mat_inv_{}'.format(frame_num)] = np.linalg.inv(world_mat)


        pcd = trimesh.load(os.path.join(work_dir, 'sparse_points_interest.ply'))

        important_pcd = trimesh.load(os.path.join(work_dir, 'sparse_points.ply'))

        vertices = pcd.vertices
        bbox_max = np.max(vertices, axis=0)
        bbox_min = np.min(vertices, axis=0)
        center = (bbox_max + bbox_min) * 0.5
        radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
        scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
        scale_mat[:3, 3] = center


        for view_num in range(n_views):
            # cam_dict['scale_mat_{}'.format(i)] = scale_mat
            cam_dict[f'view_{view_num}']['scale_mat_{}'.format(frame_num)] = scale_mat
            # cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)
            cam_dict[f'view_{view_num}']['scale_mat_inv_{}'.format(frame_num)] = np.linalg.inv(scale_mat)

        # out_dir = os.path.join(work_dir, 'preprocessed')
        # os.makedirs(out_dir, exist_ok=True)
        # os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
        # os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

        # image_list = glob(os.path.join(work_dir, 'images/*.png'))
        # image_list.sort()

        # for i, image_path in enumerate(image_list):
            # img = cv.imread(image_path)
            # cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
            # cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)



    # np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    # save aabb to a txt file

    aabb = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    np.savetxt(os.path.join(args.extracted_images_folder, 'aabb.txt'), aabb)
    for view_num in range(n_views):
        np.savez(os.path.join(args.extracted_images_folder, f'cam_{view_num}/cameras_sphere.npz'), **cam_dict[f'view_{view_num}'])
    
    
    print('Process done!')
