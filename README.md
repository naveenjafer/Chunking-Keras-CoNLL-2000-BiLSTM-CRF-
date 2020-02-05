# Chunking-Keras-CoNLL-2000-BiLSTM-CRF
This is an implementation for the Chunking task as listed under [CoNLL 2000 Dataset](https://www.clips.uantwerpen.be/conll2000/chunking/).  

## Architecture
It is a BiLstm and CRF implementation. The architecture is based off the paper titled [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf)

## Motivation
[Rohit's repo](https://github.com/rohitx007/Named-Entity-Extraction-and-Recognition) on Named Entity Extraction using movies dataset was a very good starting point for this implementation. Some of the code has been used as is in the implementation.

## Requirements
Tested with Python >= 1.7.0 & Python <= 1.15.0  
Keras 2.2.4  
Note: The code is written for a CPU implementation.

## Evaluation
The script creates auxilarry files during the run for the predicted tags of the input sentences. Although the model predicts all the tags, I have only implemented evaluation of Precision, Recall and F1Score for Noun Phrase chunks. You can easily extend this to also evaluate Verb Phrases and PPN etc. 

## Instructions to run
I have included the jupyter notebook file and the corresponding python3 vanilla file version of the same.

### Jupyter notebook
`jupyter notebook`  

### python3 script
`python3 BiLstm_+_crf_for_chunking.py`  

## Future Work
Working on making this compatible with Tensorflow > 2.0 and corresponding Keras versions.
