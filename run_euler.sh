# srun --time=1:00:00 --gpus=quadro_rtx_6000:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=7000 --tmp=2000
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 tag=128"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 tag=64"'

sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[0] tag=128ResFields0"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[1] tag=128ResFields1"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[2] tag=128ResFields2"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[3] tag=128ResFields3"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[4] tag=128ResFields4"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[5] tag=128ResFields5"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[6] tag=128ResFields6"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[7] tag=128ResFields7"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[8] tag=128ResFields8"'

sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[0] tag=64ResFields0"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[1] tag=64ResFields1"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[2] tag=64ResFields2"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[3] tag=64ResFields3"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[4] tag=64ResFields4"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[5] tag=64ResFields5"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[6] tag=64ResFields6"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[7] tag=64ResFields7"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[8] tag=64ResFields8"'
# Submitted batch job 18133107
# Submitted batch job 18133108
# Submitted batch job 18133109
# Submitted batch job 18133110
# Submitted batch job 18133111
# Submitted batch job 18133112
# Submitted batch job 18133113
# Submitted batch job 18133114
# Submitted batch job 18133115
# Submitted batch job 18133116
# Submitted batch job 18133117
# Submitted batch job 18133118
# Submitted batch job 18133119
# Submitted batch job 18133120
# Submitted batch job 18133121
# Submitted batch job 18133122
# Submitted batch job 18133123
# Submitted batch job 18133124
# Submitted batch job 18133125
# Submitted batch job 18133126

sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 tag=timeBalanced_128 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 tag=timeBalanced_64 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'

sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[0] tag=timeBalanced_128ResFields0 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[1] tag=timeBalanced_128ResFields1 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[2] tag=timeBalanced_128ResFields2 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[3] tag=timeBalanced_128ResFields3 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[4] tag=timeBalanced_128ResFields4 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[5] tag=timeBalanced_128ResFields5 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[6] tag=timeBalanced_128ResFields6 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[7] tag=timeBalanced_128ResFields7 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[8] tag=timeBalanced_128ResFields8 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'

sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[0] tag=timeBalanced_64ResFields0 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[1] tag=timeBalanced_64ResFields1 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[2] tag=timeBalanced_64ResFields2 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[3] tag=timeBalanced_64ResFields3 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[4] tag=timeBalanced_64ResFields4 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[5] tag=timeBalanced_64ResFields5 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[6] tag=timeBalanced_64ResFields6 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[7] tag=timeBalanced_64ResFields7 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'
sbatch --time=24:00:00 --gpus=1 --gpus=rtx_3090:1 --ntasks 1 --cpus-per-task=2 --mem-per-cpu=8000 --tmp=2000 --wrap='bash run2.sh TMP "python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=mv_basketball_neurips2023_10 model.sdf_net.d_hidden=64 model.sdf_net.independent_layers=[8] tag=timeBalanced_64ResFields8 model.sampling.strategy=time_balanced model.sampling.train_num_rays=1200"'

# Submitted batch job 18135286
# Submitted batch job 18135287
# Submitted batch job 18135288
# Submitted batch job 18135289
# Submitted batch job 18135290
# Submitted batch job 18135291
# Submitted batch job 18135292
# Submitted batch job 18135293
# Submitted batch job 18135294
# Submitted batch job 18135295
# Submitted batch job 18135296
# Submitted batch job 18135297
# Submitted batch job 18135298
# Submitted batch job 18135299
# Submitted batch job 18135300
# Submitted batch job 18135301
# Submitted batch job 18135302
# Submitted batch job 18135303
# Submitted batch job 18135304
# Submitted batch job 18135305

##### Video experiments
dataset.video_path=../datasets/video_data/cat_video.mp4
dataset.video_path=skvideo.datasets.bikes
# python launch.py --gpu -1 --config ./configs/video/base.yaml --train model.hidden_features=512 tag=Siren512
python launch.py --gpu -1 --config ./configs/video/base.yaml --train model.hidden_features=1024 tag=Siren1024
python launch.py --gpu -1 --config ./configs/video/base.yaml --train model.hidden_features=1700 tag=Siren1700 # 8.7M params
python launch.py --gpu -1 --config ./configs/video/base.yaml --train model.hidden_features=512 tag=Siren512ResFields123_10 model.independent_layers=[1,2,3] --composition_rank 10 # 8.7M params
python launch.py --gpu -1 --config ./configs/video/base.yaml --train model.hidden_features=512 tag=Siren512ResFields123_40 model.independent_layers=[1,2,3] --composition_rank 40 # 8.7M params
