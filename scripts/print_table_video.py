import os
import json
import pandas as pd
import numpy as np
import collections
import yaml
def parse_yaml_file(path):
    with open(path) as f:
        results_dict = yaml.safe_load(f)
    return results_dict

def format_value(key:str, val, scale=1.):
    if 'ssim' in key:
        val = round(val*100, 2)
    elif 'psnr' in key:
        val = round(val, 2)
    elif 'CD' in key:
        val = round(val*1000., 1)

    return str(val)

def main():
    # keys = ['test/rgb_psnr', 'test/full_psnr', 'test/full_ssim', 'test/mask_bce', 'test/full_lpips', 'test/depth_l1', 'CD']
    # keys = ['CD', 'test/mask_bce', 'test/full_ssim', 'test/full_psnr', 'test/full_lpips']
    # keys_str = [_k.replace('test/', '') for _k in keys]
    # keys_str = ['CD', 'Mask', 'SSIM', 'PSNR', 'LPIPS']
    ABL_EXP = True
    if ABL_EXP:
        DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_video_abl/video'
        DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_video_abl_0.3/video'
        # DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_video_ablLayerRank/video'
        # DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_video_abl_noise_rep/video'
        # DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_video_inter_rnd_repp/video'
    else:
        DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_video_rnd/video'
    keys = ['metric_test_psnr', 'metric_train_psnr']
    keys_str = ['test PSNR', 'train PSNR']
    mean_dict = collections.defaultdict(list)
    for dataset in ['cat_video.mp4', 'skvideo.datasets.bikes']:
        file_name = os.path.join('save', 'results_it100000.yaml')
        if ABL_EXP:
            exp_dir = [
            #     ('ResNet', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_resnet', file_name)),
            #     ('ResFields none', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_none_rep', file_name)),
                ('baseSiren512', os.path.join(DIR_NAME, dataset, 'baseSiren512', file_name)),
                # ('ResFields none', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_none', file_name)),
                ('ResFields cp_010', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_cp_010', file_name)),
                ('ResFields cp_020', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_cp_020', file_name)),
                ('ResFields cp_040', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_cp_040', file_name)),
                ('ResFields cp_080', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_cp_080', file_name)),
                ('ResFields tucker128_010', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker128_010', file_name)),
                ('ResFields tucker128_020', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker128_020', file_name)),
                ('ResFields tucker128_040', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker128_040', file_name)),
                ('ResFields tucker128_080', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker128_080', file_name)),
                ('ResFields tucker256_010', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker256_010', file_name)),
                ('ResFields tucker256_020', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker256_020', file_name)),
                ('ResFields tucker256_040', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker256_040', file_name)),
                ('ResFields tucker256_080', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker256_080', file_name)),
                ('ResFields tucker64_010', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker64_010', file_name)),
                ('ResFields tucker64_020', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker64_020', file_name)),
                ('ResFields tucker64_040', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker64_040', file_name)),
                ('ResFields tucker64_080', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_tucker64_080', file_name)),

                ('loe_2_4_8', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_loe_2_4_8', file_name)),
                ('loe_4_8_16', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_loe_4_8_16', file_name)),
                ('loe_8_16_32', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_loe_8_16_32', file_name)),
                ('loe_16_32_64', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_loe_16_32_64', file_name)),
                ('loe_32_64_128', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_loe_32_64_128', file_name)),

                ('ResFields vm_010', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010', file_name)),
                ('ResFields vm_020', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_020', file_name)),
                ('ResFields vm_040', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_040', file_name)),
                ('ResFields vm_080', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_080', file_name)),

            ]
            # exp_dir = [
            #     ('Siren 512 ResFields123_05', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_05', file_name)),
            #     ('Siren 512 ResFields123_10', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_10', file_name)),
            #     ('Siren 512 ResFields123_15', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_15', file_name)),
            #     ('Siren 512 ResFields123_20', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_20', file_name)),
            #     ('Siren 512 ResFields2_15', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields2_15', file_name)),
            #     ('Siren 512 ResFields2_30', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields2_30', file_name)),
            #     ('Siren 512 ResFields2_45', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields2_45', file_name)),
            #     ('Siren 512 ResFields2_60', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields2_60', file_name)),
            # ]
            # exp_dir = [
            #     ('Siren512 _noise0.1', os.path.join(DIR_NAME, dataset, 'baseSiren512_noise0.1', file_name)),
            #     ('Siren512 ResFields123_vm_010_noise0.1', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010_noise0.1', file_name)),
            #     ('Siren512 _noise0.2', os.path.join(DIR_NAME, dataset, 'baseSiren512_noise0.2', file_name)),
            #     ('Siren512 ResFields123_vm_010_noise0.2', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010_noise0.2', file_name)),
            #     ('Siren512 _noise0.3', os.path.join(DIR_NAME, dataset, 'baseSiren512_noise0.3', file_name)),
            #     ('Siren512 ResFields123_vm_010_noise0.3', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010_noise0.3', file_name)),
            #     ('Siren512 _noise0.4', os.path.join(DIR_NAME, dataset, 'baseSiren512_noise0.4', file_name)),
            #     ('Siren512 ResFields123_vm_010_noise0.4', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010_noise0.4', file_name)),
            #     ('Siren512 _noise0.5', os.path.join(DIR_NAME, dataset, 'baseSiren512_noise0.5', file_name)),
            #     ('Siren512 ResFields123_vm_010_noise0.5', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010_noise0.5', file_name)),
            #     ('Siren512 _noise0.7', os.path.join(DIR_NAME, dataset, 'baseSiren512_noise0.7', file_name)),
            #     ('Siren512 ResFields123_vm_010_noise0.7', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010_noise0.7', file_name)),

            #     ('Siren1024 _noise0.1', os.path.join(DIR_NAME, dataset, 'baseSiren1024_noise0.1', file_name)),
            #     ('Siren1024 ResFields123_vm_010_noise0.1', os.path.join(DIR_NAME, dataset, 'baseSiren1024ResFields123_vm_010_noise0.1', file_name)),
            #     ('Siren1024 _noise0.2', os.path.join(DIR_NAME, dataset, 'baseSiren1024_noise0.2', file_name)),
            #     ('Siren1024 ResFields123_vm_010_noise0.2', os.path.join(DIR_NAME, dataset, 'baseSiren1024ResFields123_vm_010_noise0.2', file_name)),
            #     ('Siren1024 _noise0.3', os.path.join(DIR_NAME, dataset, 'baseSiren1024_noise0.3', file_name)),
            #     ('Siren1024 ResFields123_vm_010_noise0.3', os.path.join(DIR_NAME, dataset, 'baseSiren1024ResFields123_vm_010_noise0.3', file_name)),
            #     ('Siren1024 _noise0.4', os.path.join(DIR_NAME, dataset, 'baseSiren1024_noise0.4', file_name)),
            #     ('Siren1024 ResFields123_vm_010_noise0.4', os.path.join(DIR_NAME, dataset, 'baseSiren1024ResFields123_vm_010_noise0.4', file_name)),
            #     ('Siren1024 _noise0.5', os.path.join(DIR_NAME, dataset, 'baseSiren1024_noise0.5', file_name)),
            #     ('Siren1024 ResFields123_vm_010_noise0.5', os.path.join(DIR_NAME, dataset, 'baseSiren1024ResFields123_vm_010_noise0.5', file_name)),
            #     ('Siren1024 _noise0.7', os.path.join(DIR_NAME, dataset, 'baseSiren1024_noise0.7', file_name)),
            #     ('Siren1024 ResFields123_vm_010_noise0.7', os.path.join(DIR_NAME, dataset, 'baseSiren1024ResFields123_vm_010_noise0.7', file_name)),
            #     ('Siren512 ResFields123_vm_010_noise0.1_mul', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010_noise0.1_modulated', file_name)),
            #     ('Siren512 ResFields123_vm_010_noise0.1_none', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm_010_noise0.1_none', file_name)),
            # ]
            # exp_dir = [
            #     ('Siren-512 ', os.path.join(DIR_NAME, dataset, 'baseSiren512', file_name)),
            #     ('lookup1.0', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_lookup1.0', file_name)),
            #     ('inter1.0', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter1.0', file_name)),
            #     ('inter0.95', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.95', file_name)),
            #     ('inter0.9', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.9', file_name)),
            #     ('inter0.8', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.8', file_name)),
            #     ('inter0.7', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.7', file_name)),
            #     ('inter0.6', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.6', file_name)),
            #     ('inter0.5', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.5', file_name)),
            #     ('inter0.4', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.4', file_name)),
            #     ('inter0.3', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.3', file_name)),
            #     ('inter0.2', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.2', file_name)),
            #     ('inter0.1', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_vm10_inter0.1', file_name)),
            # ]
        else:
            exp_dir = [
                    # ('Siren512', os.path.join(DIR_NAME, dataset, 'baseSiren512', file_name)),
                    # ('Siren512ResFields123', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123', file_name)),

                    # ('Siren1024', os.path.join(DIR_NAME, dataset, 'baseSiren1024', file_name)),
                    ('Siren1024ResFields123', os.path.join(DIR_NAME, dataset, 'baseSiren1024ResFields123', file_name)),
                    ('Siren1700', os.path.join(DIR_NAME, dataset, 'baseSiren1700', file_name)),

                    # ('ngpT20_L6', os.path.join(DIR_NAME, dataset, 'ngpT20_L6', file_name)),
                    # ('ngpT20_L7', os.path.join(DIR_NAME, dataset, 'ngpT20_L7', file_name)),
                    # ('ngpT20_L8', os.path.join(DIR_NAME, dataset, 'ngpT20_L8', file_name)),
                    # ('ngpT20_L9', os.path.join(DIR_NAME, dataset, 'ngpT20_L9', file_name)),
                    # # ('ngpT20_L10', os.path.join(DIR_NAME, dataset, 'ngpT20_L10', file_name)),
                    # ('ngpT21_L6', os.path.join(DIR_NAME, dataset, 'ngpT21_L6', file_name)),
                    # ('ngpT21_L7', os.path.join(DIR_NAME, dataset, 'ngpT21_L7', file_name)),
                    # ('ngpT21_L8', os.path.join(DIR_NAME, dataset, 'ngpT21_L8', file_name)),
                    # ('ngpT21_L9', os.path.join(DIR_NAME, dataset, 'ngpT21_L9', file_name)),
                    # # ('ngpT21_L10', os.path.join(DIR_NAME, dataset, 'ngpT21_L10', file_name)),
                    # ('ngpT22_L6', os.path.join(DIR_NAME, dataset, 'ngpT22_L6', file_name)),
                    # ('ngpT22_L7', os.path.join(DIR_NAME, dataset, 'ngpT22_L7', file_name)),
                    # ('ngpT22_L8', os.path.join(DIR_NAME, dataset, 'ngpT22_L8', file_name)),
                    # ('ngpT22_L9', os.path.join(DIR_NAME, dataset, 'ngpT22_L9', file_name)),
                    # # ('ngpT22_L10', os.path.join(DIR_NAME, dataset, 'ngpT22_L10', file_name)),
                    # ('ngpT23_L6', os.path.join(DIR_NAME, dataset, 'ngpT23_L6', file_name)),
                    # ('ngpT23_L7', os.path.join(DIR_NAME, dataset, 'ngpT23_L7', file_name)),
                    # ('ngpT23_L8', os.path.join(DIR_NAME, dataset, 'ngpT23_L8', file_name)),
                    # ('ngpT23_L9', os.path.join(DIR_NAME, dataset, 'ngpT23_L9', file_name)),
                    # # ('ngpT23_L10', os.path.join(DIR_NAME, dataset, 'ngpT23_L10', file_name)),
                    # ('ngpT24_L6', os.path.join(DIR_NAME, dataset, 'ngpT24_L6', file_name)),
                    # ('ngpT24_L7', os.path.join(DIR_NAME, dataset, 'ngpT24_L7', file_name)),
                    # ('ngpT24_L8', os.path.join(DIR_NAME, dataset, 'ngpT24_L8', file_name)),
                    # ('ngpT24_L9', os.path.join(DIR_NAME, dataset, 'ngpT24_L9', file_name)),
                    # ('ngpT24_L10', os.path.join(DIR_NAME, dataset, 'ngpT24_L10', file_name)),
            ]

        table, exp_names = [], []
        for (method_name, result_file) in exp_dir:
            results = parse_yaml_file(result_file)
            exp_names.append(method_name) # os.path.basename(os.path.dirname(result_file))
            table.append([float(format_value(k, results[k])) for k in keys])
            for k in keys:
                mean_dict[f'{method_name}#{k}'].append(results[k])
        df = pd.DataFrame(table, columns = keys_str, index=[exp_names])
        print('\n', df.to_latex(), '\n\n')
        # print(dataset.upper(), '\n', df.to_latex(), '\n\n')

    method_names = [method_name for (method_name, result_file) in exp_dir]
    mean_dict = {key: float(np.mean(val)) for key, val in mean_dict.items()}

    table = []
    for method_name in method_names:
        table.append([])
        for k in keys:
        #     table[-1].append(float(mean_dict[f'{method_name}#{k}']))
            table[-1].append(format_value(k, mean_dict[f'{method_name}#{k}']))
        # table.append([float(format_value(k, mean_dict[f'{method_name}#{k}'])) for k in keys])
    df = pd.DataFrame(table, columns = keys_str, index=[method_names])
    print('\nMEAN\n', df.to_latex(), '\n\n')

if __name__ == "__main__":    
    main()

