CUDA_VISIBLE_DEVICES=0,2 python translate.py -gpu 0 \
                    -batch_size 3 \
                    -beam_size 5 \
                    -model model_cnndm/Feb7__step_20000.pt \
		    -src /home/lily/af726/spring-2019/summarization_general/DUC_truncated_second/DUC04_concate.src \
		    -output testout/Feb7__step_20000_100.out \
                    -min_length 100 \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -verbose \
                    -block_ngram_repeat 3 \
                    -ignore_when_blocking "." "</t>" "<t>"

#-src /home/lily/zl379/Projects/Multi-doc/DUC04_PG_concate/DUC04_concate.src
