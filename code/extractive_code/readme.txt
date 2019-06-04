==
The lexrank is based on the package: https://pypi.org/project/lexrank/

The textrank is based on the https://radimrehurek.com/gensim/summarization/summariser.html

The mmr is based on the https://github.com/vishnu45/NLP-Extractive-NEWS-summarization-using-MMR/blob/master/mmr_summarizer.py

We adapted their code to accommodate on our dataset.

==

Use python 3 to run

* USAGE: python lexrank.py <dataset_path> <dataset_name> <output_file_name>

<dataset_path> is the directory where the dataset is stored, please remember / in the end.

<dataset_name> is the name of the dataset file.

<output_file_name> is the name of the output file, which contains the summarization results.