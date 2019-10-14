python train.py -save_model models/cnndm_sent \
                -data /data/projects/multi-document-summarization/OpenNMT-py-0.3.0/data/cnndm/mmr/CNNDM \
		-copy_attn -global_attention mlp -word_vec_size 128 -rnn_size 512 -layers 1 -encoder_type brnn -train_steps 20000 -max_grad_norm 2 -dropout 0. -batch_size 2 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -reuse_copy_attn -copy_loss_by_seqlength -bridge -seed 777 -world_size 1  -gpu_ranks 0 -save_checkpoint_steps 500 -train_from /home/lily/zl379/Projects/ANLP/OpenNMT_summarization/models/cnndm_sent_step_9500.pt
