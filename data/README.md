**Multi-News dataset**:

**UPDATE:**</br>
[Google Drive link](https://drive.google.com/open?id=1qZ3zJBv0zrUy4HVWxnx33IsrHGimXLPy) for preprocessed dataset.
[Link to unprocessed data](https://drive.google.com/open?id=1uDarzpu2HFc-vjXNJCRv2NIHzakpSGOw) (only replaced \n with "NEWLINE_CHAR" and append "|||||" to the end of each story. 

Please see the document `license.txt` (same as the [Newsroom Dataset](https://summari.es/download/)) for terms of use.

To reproduce the dataset (requires Python >= 3.6):

1. Download the source articles and summaries to their corresponding files (see `inputs.txt` and `summaries.txt` for a tsv of links to be downloaded and the location they should be downloaded to). We used scrapy in this project (and provide the code we used in `./scripts/scrapy/`), although you may try using wget or curl. 
2. Run `prep_data.py` from the `./scripts/` folder to clean, tokenize and split data into train, validation and test sets. 



**File structure**: 

`./inputs.txt `-- a tab-separated file where the first column is the Wayback link to be scraped and the second column is the output file the link should be scraped to for further processing. These links correspond to input source articles which were available on Wayback at the time of my scraping. As an example, a file which will be stored in `articles/136951-0.html` is an input file corresponding to `summaries/136951`. 

`./summaries.txt` -- a tab-separated file where the first column is the Wayback link to be scraped and the second column is the output file the link should be scraped to for further processing. These links correspond to gold summaries.

`./availability_checks.txt` -- a tab-separated file where the first column is the Wayback link to be scraped and the second column is the output file the link should be scraped to for further processing. These links correspond to links to Wayback's availability [API](https://archive.org/help/wayback_api.php), which may be useful to check the availability of Wayback links (if availability has changed).


`./ids/all.id` -- a file of ids corresponding to the summary ids of examples used in our experiments

`./ids/train.id` -- a file of indices which correspond to lines in `./ids/all.id` whose ids make up the training set (0-based).

`./ids/val.id` -- a file of indices which correspond to lines in `./ids/all.id` whose ids make up the validation set (0-based).

`./ids/test.id` -- a file of indices which correspond to lines in `./ids/all.id` whose ids make up the test set (0-based).

`./ids/id2sources.txt` -- a tab-separated file in which the first column corresponds to the unique summary id and the second column corresponds to associated input documents. 


**Scripts (all code uses Python >=3.6)**:

`./scripts/scrapy_scripts/` -- these corresponds to scripts to Scrapy scripts used to crawl this dataset. You may have to change the filepaths in this file: `scripts/scrapy_scripts/tutorial/spiders/quotes_spiders.py`. Additionally, I made use of [Crawlera](https://scrapinghub.com/crawlera) during this project to help with quicker scraping.  

`./scripts/prep_data.py` -- provides functions to extract main text from archive.org links, remove unwanted social media items from downloaded text, as well as check the availability of files downloaded from Wayback's availability API.

