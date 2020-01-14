python train.py \
    -data data/demo/demo \
    -save_model models/demo-model \
    -world_size 1 \
    -gpu_ranks 0 \
    -batch_size 20 \
    -report_every 25 \
    -tensorboard \
    -tensorboard_log_dir tb_logs/demo1 \
    -my_shuffle
