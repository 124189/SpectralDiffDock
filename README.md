# Diffdock_pro


初始训练指令：python train.py --run_name big_score_model --test_sigma_intervals --pdbbind_esm_embeddings_path ../data/esm2_3billion_embeddings.pt --log_dir workdir --lr 1e-3 --tr_sigma_min 0.1 --tr_sigma_max 19 --rot_sigma_min 0.03 --rot_sigma_max 1.55 --batch_size 16 --ns 48 --nv 10 --num_conv_layers 6 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --num_dataloader_workers 1 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 500 --use_ema --distance_embed_dim 64 --cross_distance_embed_dim 64 --sigma_embed_dim 64 --scheduler_patience 30 --n_epochs 850 --limit_complexes 1000 --pdbbind_dir ../data/PDBBind_processed/

加入多gpu
torchrun --nproc_per_node=4 train.py --distributed --run_name big_score_model_ddp_4gpu --test_sigma_intervals --pdbbind_esm_embeddings_path ../data/esm2_3billion_embeddings.pt --log_dir workdir --lr 1e-3 --tr_sigma_min 0.1 --tr_sigma_max 19 --rot_sigma_min 0.03 --rot_sigma_max 1.55 --batch_size 16 --ns 48 --nv 10 --num_conv_layers 6 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --num_dataloader_workers 1 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 500 --use_ema --distance_embed_dim 64 --cross_distance_embed_dim 64 --sigma_embed_dim 64 --scheduler_patience 30 --n_epochs 850 --limit_complexes 1000 --pdbbind_dir ../data/PDBBind_processed/

