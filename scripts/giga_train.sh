python train.py -save_model models \ #ok
    -data data/giga/processed \ #ok
    -copy_attn \ #ok
    -global_attention mlp \ #ok
    -word_vec_size 128 \ #ok
    -rnn_size 256 \ #ok
    -layers 2 \ #ok
    -encoder_type brnn \ #ok
    -train_steps 200000 \ #ok
    -max_grad_norm 2 \ #ok
    -dropout 0\.1 \ #ok
    -batch_size 16 \ #ok
    -valid_batch_size 16 \ #ok
    -optim adagrad \ #ok
    -learning_rate 0.15 \ #ok
    -adagrad_accumulator_init 0.1 \ #ok
    -reuse_copy_attn \ #ok
    -copy_loss_by_seqlength \ #ok
    -bridge \ #ok
    -seed 777 \  #ok
    -world_size 1 \ #ok
    -gpu_ranks 0
