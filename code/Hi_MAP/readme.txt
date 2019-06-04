==
The original code is from OpenNMT:http://opennmt.net/

We adapted their code and implemented our Hi-MAP model.

==
Main changes:

Preprocessing in onmt/inputters/text_dataset.py: we extend one more field.

MMR scores in onmt/decoders/decoder.py: add _run_mmr_attention() function.

Sentence level encoder in onmt/encoders/rnn_encoder.py

==

Use python 3 to run


* Pre-processing

Check run_prep_newser.sh

* Train

Check run_train_newser.sh

* inference

Check run_inference_newser.sh
1. vim run_inference_newser.sh
2. change the model name (number of steps)

The result will be a large file, each line is a record.

Calculate rouge score by your own.
