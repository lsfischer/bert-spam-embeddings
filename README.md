# BERT-embedded SMS spam messages

The output dataset is an extention of the existing input dataset retrieved from the [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

This repo stores the input dataset, the dataset with the embeddings and the code used to generate this dataset.

## Input Dataset Structure
The original dataset contains 5574 english messages each labelled as *spam* or *ham*
This dataset contains 4 columns:

- `v1` -> Target column specifying if the message is *spam* or *ham*
- `v2` -> The original unprocessed messages
- `Unnamed_col_1` & `Unnamed_col_2` -> Columns with mostly missing values (around 99%) that are discarded

## Encoded Dataset Structure

The output encoded dataset contains the same information as the input dataset plus the additional DiltilBERT classification embeddings. This results in a dataset with 770 columns:

- `spam` -> Target column specifying if the message is *spam* or *ham*
- `original_message` -> The original unprocessed messages
- `0` up to `768` -> columns containing the DistilBERT classification embeddings for the message, after it being processed

## Procedure

HuggingFace's DistilBERT is used from their [transformers](https://github.com/huggingface/transformers) package.

[Jay Allamar's tutorial](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) is followed to encode the messages using DistilBERT.

For memory efficiency reasons all messages are first stripped from punctuation and then english stopwords are removed. Then only the first 30 tokens are kept.

As per [my analysis](https://www.kaggle.com/mrlucasfischer/bert-the-spam-detector-that-uses-just-10-words) of this dataset on kaggle it can be seen that most *ham* messages have around 10 words and *spam* messages around 29 words, without stopwords. This means that once stopwords are removed from the messages, keeping the first 30 tokens might mean some information loss but not to critical. (Acrually in [my analysis](https://www.kaggle.com/mrlucasfischer/bert-the-spam-detector-that-uses-just-10-words) it is demonstrated that encoding the messages using only the first 10 tokens after processing them is enough to have a good encoding capable of achieving 88.1 ROC-AUC with a baseline random forest.)

## Acknowledgements

[Jay Allamar's tutorial](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) is followed to encode the messages using DistilBERT.

The original dataset is part of the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/index.php) and can be found [here](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

UCI Machine Learning urges to if you find the original dataset useful, cite the original authors found [here](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/).

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results.  Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011