CUDA_VISIBLE_DEVICES=0 python train.py -save_model irene_model/cnndm_ \
                -data data_test/cnndm_preprocessed/cnndm_sent \
                -copy_attn \
                -global_attention mlp \
                -word_vec_size 128 \
                -rnn_size 512 \
                -layers 1 \
                -encoder_type brnn \
                -train_steps 200000 \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 16 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
                -bridge \
                -seed 777 \
                -world_size 1 \
                -gpu_ranks 0 \
                -save_checkpoint_steps 1000


