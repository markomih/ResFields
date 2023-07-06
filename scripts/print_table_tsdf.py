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
        val = round(val*1000., 3)
    elif 'ND' in key:
        val = round(val*100., 3)

    return str(val)

def main():
    # keys = ['test/rgb_psnr', 'test/full_psnr', 'test/full_ssim', 'test/mask_bce', 'test/full_lpips', 'test/depth_l1', 'CD']
    # keys = ['CD', 'test/mask_bce', 'test/full_ssim', 'test/full_psnr', 'test/full_lpips']
    # keys_str = [_k.replace('test/', '') for _k in keys]
    # keys_str = ['CD', 'Mask', 'SSIM', 'PSNR', 'LPIPS']
    keys = ['metric_CD', 'metric_ND']
    keys_str = ['CD', 'ND']
    BASIC_EXP = True
    if BASIC_EXP:
        DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/'
    else:
        DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_tsdf_inter/tsdf'
    mean_dict = collections.defaultdict(list)
    # for dataset in ['bear3EP_Agression.anime', 'tigerD8H_Swim17.anime', 'vampire_Breakdance1990.anime', 'vanguard_JoyfulJump.anime']:
    for dataset in ['bear3EP_Agression.anime', 'tigerD8H_Swim17.anime', 'vampire_Breakdance1990.anime', 'vanguard_JoyfulJump.anime', 'resynth']:
    # for dataset in ['tigerD8H_Swim17.anime', 'vampire_Breakdance1990.anime', 'vanguard_JoyfulJump.anime']:
    # for dataset in ['vampire_Breakdance1990.anime']:
        file_name = os.path.join('save', 'results_it200000-test.yaml')
        if BASIC_EXP:
            exp_dir = [
                # ('ngp', os.path.join(DIR_NAME, dataset, 'ngp', file_name)),
                ('Siren-128', os.path.join(DIR_NAME, dataset, 'baseSiren128', file_name)),
                ('Siren-128 + ResFields (rank=05)', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_05', file_name)),
                ('Siren-128 + ResFields (rank=10)', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_10', file_name)),
                # ('Siren-128 + ResFields (rank=15)', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_15', file_name)),
                ('Siren-128 + ResFields (rank=20)', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20', file_name)),
                ('Siren-128 + ResFields (rank=40)', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_40', file_name)),
                ('Siren-256', os.path.join(DIR_NAME, dataset, 'baseSiren256', file_name)),
                ('Siren-256 + ResFields (rank=05)', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_05', file_name)),
                ('Siren-256 + ResFields (rank=10)', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_10', file_name)),
                # ('Siren-256 + ResFields (rank=15)', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_15', file_name)),
                ('Siren-256 + ResFields (rank=20)', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20', file_name)),
                ('Siren-256 + ResFields (rank=40)', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_40', file_name)),
                # ('Siren-512', os.path.join(DIR_NAME, dataset, 'baseSiren512', file_name)),
                # ('Siren-512 + ResFields (rank=05)', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_05', file_name)),
                # ('Siren-512 + ResFields (rank=10)', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_10', file_name)),
                # ('Siren-512 + ResFields (rank=15)', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_15', file_name)),
                # ('Siren-512 + ResFields (rank=20)', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_20', file_name)),
                # ('Siren-512 + ResFields (rank=40)', os.path.join(DIR_NAME, dataset, 'baseSiren512ResFields123_40', file_name)),
            ]
        else:
            exp_dir = [
                ('ngp', os.path.join(DIR_NAME, dataset, 'ngp', file_name)),
                ('Siren-128', os.path.join(DIR_NAME, dataset, 'baseSiren128', file_name)),
                ('Siren128ResFields123_20_frac0.1', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.1', file_name)),
                ('Siren128ResFields123_20_frac0.2', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.2', file_name)),
                ('Siren128ResFields123_20_frac0.3', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.3', file_name)),
                ('Siren128ResFields123_20_frac0.4', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.4', file_name)),
                ('Siren128ResFields123_20_frac0.5', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.5', file_name)),
                ('Siren128ResFields123_20_frac0.6', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.6', file_name)),
                ('Siren128ResFields123_20_frac0.7', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.7', file_name)),
                ('Siren128ResFields123_20_frac0.8', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.8', file_name)),
                ('Siren128ResFields123_20_frac0.9', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.9', file_name)),
                ('Siren128ResFields123_20_frac0.95', os.path.join(DIR_NAME, dataset, 'baseSiren128ResFields123_20_frac0.95', file_name)),

                # ('Siren-256', os.path.join(DIR_NAME, dataset, 'baseSiren256', file_name)),
                # ('Siren256ResFields123_20_frac0.1', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.1', file_name)),
                # ('Siren256ResFields123_20_frac0.2', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.2', file_name)),
                # ('Siren256ResFields123_20_frac0.3', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.3', file_name)),
                # ('Siren256ResFields123_20_frac0.4', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.4', file_name)),
                # ('Siren256ResFields123_20_frac0.5', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.5', file_name)),
                # ('Siren256ResFields123_20_frac0.6', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.6', file_name)),
                # ('Siren256ResFields123_20_frac0.7', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.7', file_name)),
                # ('Siren256ResFields123_20_frac0.8', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.8', file_name)),
                # ('Siren256ResFields123_20_frac0.9', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.9', file_name)),
                # ('Siren256ResFields123_20_frac0.95', os.path.join(DIR_NAME, dataset, 'baseSiren256ResFields123_20_frac0.95', file_name)),
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
