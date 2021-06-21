
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="https://github.com/hetpandya/textgenie/raw/main/logo.png" alt="logo" width="70%" />
</p>

# TextGenie

TextGenie is a python library that helps you augment your text dataset and generate similar kind of samples, thus generating a more robust dataset to train better models. It also takes care of labeled datasets while generating similar samples keeping their labels in memory. 

It uses various Natural Language Processing methods such as paraphrase generation, BERT mask filling and converting text to active voice if found in passive voices. This library currently supports `English` Language.

## Installation
```
$ pip install textgenie
```

## Example
```python
from textgenie import TextGenie

textgenie = TextGenie("ramsrigouthamg/t5_paraphraser",'bert-base-uncased')

# Augment a list of sentences
sentences = ["The video was posted on Facebook by Alex.","I plan to run it again this time"]
textgenie.magic_lamp(sentences,"paraphrase: ",n_mask_predictions=5,convert_to_active=True)

# Augment data in a txt file
textgenie.magic_lamp("sentences.txt","paraphrase: ",n_mask_predictions=5,convert_to_active=True)

# Augment data in a csv file with labels
textgenie.magic_lamp("sentences.csv","paraphrase: ",n_mask_predictions=5,convert_to_active=True)
```
## Usage
<!--ts-->
- Initializing the augmentor:

  ```textgenie = TextGenie(paraphrase_model_name='model_name',mask_model_name='model_name',spacy_model_name="model_name",device="cpu")```
  - Parameters:
    - *paraphrase_model_name*: 
      - The name of the T5 paraphrase model.
    - *mask_model_name*:
      - BERT model that will be used to fill masks. This model is disabled by default. But can be enabled by mentioning the name of the BERT model to be used. A list of mask filling models can be found [here](https://huggingface.co/models?filter=en&pipeline_tag=fill-mask)
    - *spacy_model_name*:
      - Name of the Spacy model. Available models can be found [here](https://spacy.io/models). The default value is set to *en*.
    - *device*:
      - The device where the model will be loaded. The default value is set to *cpu*.
- Methods:
  - augment_sent_mask_filling():
    - Generate augmented data using BERT mask filling. 
  - augment_sent_t5():
    - Generate augmented data using T5 paraphrasing model. 
  - convert_to_active():
    - Converts a sentence to active voice, if found in passive voice. Otherwise returns the same sentence.
  - magic_once():
    - This is a wrapper method for *augment_sent_mask_filling()*, *augment_sent_t5()* and *convert_to_active()* methods. Using this, a sentence can be augmented using all the above mentioned techniques.
  - magic_lamp():
    - This method can be used for augmenting whole dataset. Currently accepted dataset formats are: `txt`,`csv`,`tsv` and `list`. 
    - If the dataset is in `list` or `txt` format, a list of augmented sentences will be returned. Also, a `txt` file with the name *sentences_aug.txt* is saved containing the output of the augmented data. 
    - If a dataset is in `csv` or `tsv` format with labels, the dataset will be augmented along with keeping in memory the labels for the new samples and a pandas dataframe of the augmented data will be returned. A `csv` file will be generated with the augmented output with name `original_csv_file_name_aug.csv` 
<!--te-->

## References
[Passive To Active](https://github.com/DanManN/pass2act)

## License
Please check `LICENSE` for more details.

