from collections import defaultdict
# from tqdm import tqdm
import numpy as np
import os
# import imageio
from collections import defaultdict
import matplotlib.pyplot as plt
import glob
import cv2
from copy import deepcopy
# import cmapy
# import skimage
# import pathlib
import pandas as pd
# from tensorboard.backend.event_processing import event_accumulator
# from moviepy.editor import VideoFileClip, clips_array, vfx, ImageSequenceClip, CompositeVideoClip, concatenate_videoclips, VideoClip, TextClip
# from matplotlib import animation
import yaml
from matplotlib import animation

def parse_yaml_file(path):
    with open(path) as f:
        results_dict = yaml.safe_load(f)
    # results_dict = {key: round(val, 2) for key, val in results_dict.items()}
    return results_dict

def load_history(dir_path, dataset, method):
    _path = os.path.join(dir_path, dataset, method, 'save', 'results_it*.yaml')
    _path = sorted(glob.glob(_path))

    table_test, table_train = [], []
    it_list = []
    for file in _path:
        it_list.append(int(file.split('it')[-1].split('.')[0]))
        results = parse_yaml_file(file)
        table_test.append(results['metric_test_psnr'])
        table_train.append(results['metric_train_psnr'])

    return np.array(it_list), np.array(table_test), np.array(table_train)


def load_file(file_path):
    inter, psnr = [], []
    with open(file_path) as f:
        for line in f:
            _inter, _psnr = line.strip().split('\t')
            inter.append(int(_inter.strip()))
            psnr.append(float(_psnr.strip()))
    return inter, psnr

def animated_line_plot(x_axis, data:dict, trgt_path, style_dict, anim_keys=[], plot_keys=[], legend_loc='lower right', plot_type=None, use_legend=True, ylimit=None, distable_yaxis=False):
    """
    Args:
        x_axis: x-axis values
        data: dict of {label: list of y-axis values}
        trgt_path: path to save the animation
    """
    data = deepcopy(data)
    fig, ax = plt.subplots()
    fontdict = {'size': 16}
    ax.tick_params(axis='both', which='major', direction='in', labelsize=11)
    # ax.set_xlabel("Iterations", fontdict=fontdict)
    if plot_type == 'image':
        ax.set_xticks([5000, 10000, 15000])
        ax.set_xticklabels(['5,000', '10,000', '15,000'])
        ax.set_xlim(0, 15000)
        ax.set_yticks([10, 20, 30, 40, 50, 60])
        ax.set_ylim(0, 60)
    elif plot_type == 'poisson':
        ax.set_xticks([1000, 2000, 3000, 4000])
        ax.set_xticklabels(['1,000', '2,000', '3,000', '4,000'])
        ax.set_xlim(0, 4000)
        ax.set_yticks([5, 10, 15, 20, 25, 30, 35])
        ax.set_ylim(0, 35)
    ax.grid()
    if ylimit is not None:
        ax.set_ylim(ylimit)
    if distable_yaxis:
        plt.yticks(color='w')
    # else:
    #     ax.set_ylabel("PSNR", fontdict=fontdict)

    plt.yticks(color='w'); plt.xticks(color='w')

    plot_data = {key: data[key] for key in anim_keys}
    lines = []
    for key, y in plot_data.items():
        lobj = ax.plot(x_axis, y, **style_dict[key])[0]
        lines.append(lobj)

    if use_legend:
        # ax.legend(fontsize="18")
        ax.legend(loc=legend_loc, bbox_to_anchor=(0.95, 0.035))

    def update(num, x, data, lines):
        for idx, (_key, y) in enumerate(data.items()):
            lines[idx].set_data(x[:num], y[:num])
        return lines
    # plot static 
    for key in plot_keys:
        ax.plot(x_axis, data[key], **style_dict[key])
    # plot animation 
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    anim = animation.FuncAnimation(fig, update, fargs=[x_axis, plot_data, lines],
                                   frames=len(x_axis), interval=1, blit=True)
    anim.save(trgt_path, writer=writer)


def make_plot(iterations, psnrs:dict, trgt_path, style, ylabel='PSNR', xlabel='Training steps', use_legend=True, ylimit=None, distable_yaxis=False):
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', direction='in', labelsize=8)
    # ax.set_ylabel(ylabel)
    # ax.set_xlabel(xlabel)
    ax.grid()
    for key, val in psnrs.items():
        ax.plot(iterations, val, **style[key])
    if ylimit is not None:
        ax.set_ylim(ylimit)
    # X-axis tick label
    # plt.xticks(color='w')
    plt.xticks(fontsize="14")
    # Y-axis tick label
    if distable_yaxis:
        plt.yticks(color='w')
    plt.yticks(fontsize="14")

    if use_legend:
        ax.legend(fontsize="18")
    # fig.savefig(trgt_path, bbox='tight', bbox_inches='tight', pad_inches=0.)
    fig.savefig(trgt_path, bbox_inches='tight', pad_inches=0.)
    # print('Saved', trgt_path)
    return cv2.imread(trgt_path)

def cat_videos(video1, video2, trgt_path):
    cmd = f"ffmpeg -i {video1} -i {video2} -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac  -filter_complex hstack=inputs=2  {trgt_path} -y"
    os.system(cmd)

def main():
    style_dict = {}
    DIR_NAME = '/home/marko/remote_euler/projects/ResFields/exp_video_rnd/video'
    datasets = ['cat_video.mp4', 'skvideo.datasets.bikes']
    methods = ['baseSiren1024', 'baseSiren512ResFields123', 'ngpT23_L8']
    iters, train_psnr_dict, test_psnr_dict = get_results(DIR_NAME, datasets, methods)
    style_dict = {
        'baseSiren1024': dict(label='Siren-1024', color='blue', markersize=12, linestyle='dashed', linewidth=2),
        'baseSiren512ResFields123': dict(label='Siren-512+ResFields', color='black', markersize=12, linestyle='solid', linewidth=2),
        'ngpT23_L8': dict(label='NGP', color='green', markersize=12, linestyle='dashed', linewidth=2),
    }
    min_psnr = np.stack(list(train_psnr_dict.values()) + list(test_psnr_dict.values())).min()
    max_psnr = np.stack(list(train_psnr_dict.values()) + list(test_psnr_dict.values())).max()
    iterations = np.array(iters)
    ylimit = None
    ylimit = min_psnr - 0.5,  max_psnr + 0.5 #[21, 40]
    train_plt = make_plot(iterations, train_psnr_dict, './tmp_train_psnr.pdf', style_dict, ylabel='Train PSNR', xlabel='Training steps', use_legend=False, ylimit=ylimit)
    train_plt = make_plot(iterations, train_psnr_dict, './tmp_train_psnr.png', style_dict, ylabel='Train PSNR', xlabel='Training steps', use_legend=False, ylimit=ylimit)
    val_plt = make_plot(iterations, test_psnr_dict, './tmp_val_psnr.pdf', style_dict, ylabel='Test PSNR', xlabel='Training steps', use_legend=True, ylimit=ylimit, distable_yaxis=True)
    val_plt = make_plot(iterations, test_psnr_dict, './tmp_val_psnr.png', style_dict, ylabel='Test PSNR', xlabel='Training steps', use_legend=True, ylimit=ylimit, distable_yaxis=True)

    cv2.imwrite('vide_exp.png', np.concatenate((train_plt, val_plt[:train_plt.shape[0], :train_plt.shape[1]]), axis=1))
    print('vide_exp.png')
    name_mapping = {key: val['label'] for key, val in style_dict.items()}
    
    # plot animated Siren1024
    animated_line_plot(iterations, train_psnr_dict, './tmp_train_psnr.mp4', style_dict, anim_keys=['baseSiren1024'], plot_keys=[], use_legend=False, ylimit=ylimit, distable_yaxis=False)
    animated_line_plot(iterations, test_psnr_dict, './tmp_test_psnr.mp4', style_dict, anim_keys=['baseSiren1024'], plot_keys=[], use_legend=False, ylimit=ylimit, distable_yaxis=True)
    cat_videos('./tmp_train_psnr.mp4', './tmp_test_psnr.mp4', './anim1.mp4')

    # plot static Siren1024, animated Siren512ResFields123
    animated_line_plot(iterations, train_psnr_dict, './tmp_train_psnr.mp4', style_dict, anim_keys=['baseSiren512ResFields123'], plot_keys=['baseSiren1024'], use_legend=False, ylimit=ylimit, distable_yaxis=False)
    animated_line_plot(iterations, test_psnr_dict, './tmp_test_psnr.mp4', style_dict, anim_keys=['baseSiren512ResFields123'], plot_keys=['baseSiren1024'], use_legend=False, ylimit=ylimit, distable_yaxis=True)
    cat_videos('./tmp_train_psnr.mp4', './tmp_test_psnr.mp4', './anim2.mp4')

    # plot static Siren1024 and Siren512ResFields123, animated NGP
    animated_line_plot(iterations, train_psnr_dict, './tmp_train_psnr.mp4', style_dict, anim_keys=['ngpT23_L8'], plot_keys=['baseSiren1024', 'baseSiren512ResFields123'], use_legend=False, ylimit=ylimit, distable_yaxis=False)
    animated_line_plot(iterations, test_psnr_dict, './tmp_test_psnr.mp4', style_dict, anim_keys=['ngpT23_L8'], plot_keys=['baseSiren1024', 'baseSiren512ResFields123'], use_legend=False, ylimit=ylimit, distable_yaxis=True)
    cat_videos('./tmp_train_psnr.mp4', './tmp_test_psnr.mp4', './anim3.mp4')

    print('Saved', './anim3.mp4')


def get_results(dir_name, datasets, methods):
    to_ret_train = dict()
    to_ret_test = dict()
    iters = None
    for method in methods:
        data_list_test, data_list_train = [], []
        for dataset in datasets:
            _it, data_test, data_train = load_history(dir_name, dataset, method)
            if iters is None:
                iters = _it
            else:
                assert np.all(iters == _it)
            
            data_list_test.append(data_test)
            data_list_train.append(data_train)

        to_ret_train[method] = np.stack(data_list_train).mean(0)
        to_ret_test[method] = np.stack(data_list_test).mean(0)

    return iters, to_ret_train, to_ret_test

if __name__ == '__main__':
    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    main()
