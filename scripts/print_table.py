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
    keys = ['metric_CD', 'coarse_metric_ssim', 'coarse_metric_psnr']
    keys_str = ['CD', 'SSIM', 'PSNR']
    mean_dict = collections.defaultdict(list)
    DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf'
    DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k_nerf/dysdf'
    for dataset in ['mv_basketball_neurips2023_10', 'model', 'dancer_vox11', 'exercise_vox11']:
        file_name = os.path.join('save', 'results_it400000-test.yaml')
        exp_dir = [
            ('TNeRF', os.path.join(DIR_NAME, dataset, 'tnerf', file_name)),
            ('TNeRF + ResFields1', os.path.join(DIR_NAME, dataset, 'tnerfResFields1', file_name)),
            # ('TNeRF + ResFields12', os.path.join(DIR_NAME, dataset, 'tnerfResFields12', file_name)),
            ('TNeRF + ResFields123', os.path.join(DIR_NAME, dataset, 'tnerfResFields123', file_name)),
            ('TNeRF + ResFields1234567', os.path.join(DIR_NAME, dataset, 'tnerfResFields1234567', file_name)),

            ('DyNeRF', os.path.join(DIR_NAME, dataset, 'dynerf', file_name)),
            ('DyNeRF + ResFields1', os.path.join(DIR_NAME, dataset, 'dynerfResFields1', file_name)),
            # ('DyNeRF + ResFields12', os.path.join(DIR_NAME, dataset, 'dynerfResFields12', file_name)),
            ('DyNeRF + ResFields123', os.path.join(DIR_NAME, dataset, 'dynerfResFields123', file_name)),
            ('DyNeRF + ResFields1234567', os.path.join(DIR_NAME, dataset, 'dynerfResFields1234567', file_name)),

            ('DNeRF', os.path.join(DIR_NAME, dataset, 'dnerf', file_name)),
            ('DNeRF + ResFields1', os.path.join(DIR_NAME, dataset, 'dnerfResFields1', file_name)),
            # ('DNeRF + ResFields12', os.path.join(DIR_NAME, dataset, 'dnerfResFields12', file_name)),
            ('DNeRF + ResFields123', os.path.join(DIR_NAME, dataset, 'dnerfResFields123', file_name)),
            ('DNeRF + ResFields1234567', os.path.join(DIR_NAME, dataset, 'dnerfResFields1234567', file_name)),

            ('Nerfies', os.path.join(DIR_NAME, dataset, 'nerfies', file_name)),
            ('Nerfies + ResFields1', os.path.join(DIR_NAME, dataset, 'nerfiesResFields1', file_name)),
            # ('Nerfies + ResFields12', os.path.join(DIR_NAME, dataset, 'nerfiesResFields12', file_name)),
            ('Nerfies + ResFields123', os.path.join(DIR_NAME, dataset, 'nerfiesResFields123', file_name)),
            ('Nerfies + ResFields1234567', os.path.join(DIR_NAME, dataset, 'nerfiesResFields1234567', file_name)),

            ('HyperNeRF', os.path.join(DIR_NAME, dataset, 'hypernerf', file_name)),
            ('HyperNeRF + ResFields1', os.path.join(DIR_NAME, dataset, 'hypernerfResFields1', file_name)),
            # ('HyperNeRF + ResFields12', os.path.join(DIR_NAME, dataset, 'hypernerfResFields12', file_name)),
            ('HyperNeRF + ResFields123', os.path.join(DIR_NAME, dataset, 'hypernerfResFields123', file_name)),
            ('HyperNeRF + ResFields1234567', os.path.join(DIR_NAME, dataset, 'hypernerfResFields1234567', file_name)),

            ('NDR', os.path.join(DIR_NAME, dataset, 'ndr', file_name)),
            ('NDR + ResFields1', os.path.join(DIR_NAME, dataset, 'ndrResFields1', file_name)),
            # ('NDR + ResFields12', os.path.join(DIR_NAME, dataset, 'ndrResFields12', file_name)),
            ('NDR + ResFields123', os.path.join(DIR_NAME, dataset, 'ndrResFields123', file_name)),
            ('NDR + ResFields1234567', os.path.join(DIR_NAME, dataset, 'ndrResFields1234567', file_name)),
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
