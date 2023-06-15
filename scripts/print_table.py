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
    for dataset in ['mv_basketball_neurips2023_10', 'model', 'dancer_vox11', 'exercise_vox11']:
        file_name = 'results_it400000-test.yaml'
        exp_dir = [
            ('TNeRF', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/tnerf/save/{file_name}'),
            ('TNeRF + ResFields', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/tnerfResFields1/save/{file_name}'),

            ('DyNeRF', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/dynerf/save/{file_name}'),
            ('DyNeRF + ResFields', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/dynerfResFields1/save/{file_name}'),

            ('DNeRF', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/dnerf/save/{file_name}'),
            ('DNeRF + ResFields', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/dnerfResFields1/save/{file_name}'),

            ('Nerfies', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/nerfies/save/{file_name}'),
            ('Nerfies + ResFields', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/nerfiesResFields1/save/{file_name}'),

            ('HyperNeRF', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/hypernerf/save/{file_name}'),
            ('HyperNeRF + ResFields', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/hypernerfResFields1/save/{file_name}'),

            ('NDR', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/ndr/save/{file_name}'),
            ('NDR + ResFields', f'/home/marko/remote_euler/projects/ResFields/exp_owlii_benchmarking400k/dysdf/{dataset}/ndrResFields1/save/{file_name}'),
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
