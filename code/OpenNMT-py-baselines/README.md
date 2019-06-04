# BRNN & Transformer baseline

### Codes Structure
We obtained the baseline results of BRNN and Transformer from OpenNMT implementation. For detailed documentation of OpenNMT, please refer to [this](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/Summarization.md) page. Therefore, this folder mainly contains the OpenNMT codes with our running scripts.

### Environments
To replicate the environments for runninng OpenNMT, simply run 
```sh
$ conda env create -f environment.yml
$ conda activate OpenNMT
```

### Preprocess
To get the training and validation data in OpenNMT format, you first need to preprocess the input data. The raw data should be in text files with one record per line. You need to modify the path to text files in the `run_preprocess.sh` before running the script.
```sh
$ ./run_preprocess.sh
```

### Training
We used the same training parameters as on OpenNMT Summarization page. You can modify the `run_train_brnn/transformer.sh` to use different parameter settings. If training is interrupted, you can add `-train_from` with model path to continue training.
```sh
$ ./run_train_brnn.sh
$ ./run_train_transformer.sh
```
We also uploaded our trained BRNN and Transformer models in the `models\` directory.


### Testing
To test the model on test data, run the `run_test_brnn/transformer.sh` script. You need to modify the path to the source text file of test data (in the format of one example per line).
```sh
$ ./run_test_brnn.sh
$ ./run_test_transformer.sh
```
These 2 scripts contains our testing parameters on Multi-News data. If you want to replicate our results on DUC data, you should modify the minimum length and maximum lenght of decoding to 50 and 100 respectively.

