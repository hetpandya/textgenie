
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://static.pepy.tech/badge/textgenie)](https://pepy.tech/project/textgenie)

<p align="center">
  <img src="https://github.com/hetpandya/textgenie/raw/main/logo.png" alt="logo" width="70%" />
</p>

# TextGenie

TextGenie is a text data augmentations library that helps you augment your text dataset and generate similar kind of samples, thus generating a more robust dataset to train better models. It also takes care of labeled datasets while generating similar samples keeping their labels in memory. 

It uses various Natural Language Processing methods such as paraphrase generation, BERT mask filling and converting text to active voice if found in passive voices. This library currently supports `English` Language.

## Installation
```
pip install textgenie
```

## Example
```python
from textgenie import TextGenie

textgenie = TextGenie("hetpandya/t5-small-tapaco", "bert-base-uncased")

# Augment a list of sentences
sentences = [
    "The video was posted on Facebook by Alex.",
    "I plan to run it again this time",
]
textgenie.magic_lamp(
    sentences, "paraphrase: ", n_mask_predictions=5, convert_to_active=True
)

# Augment data in a txt file
textgenie.magic_lamp(
    "sentences.txt", "paraphrase: ", n_mask_predictions=5, convert_to_active=True
)

# Augment data in a csv file with labels
textgenie.magic_lamp(
    "sentences.csv",
    "paraphrase: ",
    n_mask_predictions=5,
    convert_to_active=True,
    label_column="Label",
    data_column="Text",
    column_names=["Text", "Label"],
)
```
Examples can be found in the examples [notebook](https://github.com/hetpandya/textgenie/blob/main/examples/examples.ipynb).

## Usage
<!--ts-->
- Initializing the augmentor:
  ```textgenie = TextGenie(paraphrase_model_name='model_name',mask_model_name='model_name',spacy_model_name="model_name",device="cpu")```
  - Parameters:
    - *paraphrase_model_name*: 
      - The name of the T5 paraphrase model.
      - A list of pretrained model for paraphrase generation can be found [here](https://github.com/hetpandya/paraphrase-datasets-pretrained-models#pretrained-models)
    - *mask_model_name*:
      - BERT model that will be used to fill masks. This model is disabled by default. But can be enabled by mentioning the name of the BERT model to be used. A list of mask filling models can be found [here](https://huggingface.co/models?filter=en&pipeline_tag=fill-mask)
    - *spacy_model_name*:
      - Name of the Spacy model. Available models can be found [here](https://spacy.io/models). The default value is set to *en*.
    - *device*:
      - The device where the model will be loaded. The default value is set to *cpu*.
- Methods:  
  - augment_sent_mask_filling():
    - Generate augmented data using BERT mask filling.
    - Parameters:
      - *sent*:
        - The sentence on which augmentation has to be applied.
      - *n_mask_predictions*:  
        - The number of predictions, the BERT mask filling model should generate. The default value is set to *5*.
  - augment_sent_t5():
    - Generate augmented data using T5 paraphrasing model. 
    - Parameters:
      - *sent*:
        - The sentence on which augmentation has to be applied. 
      - *prefix*:
        - The prefix for the T5 model input.
      - *n_predictions*:
        - The number of number augmentations, the function should return. The default value is set to *5*.
      - *top_k*:
        - The number of predictions, the T5 model should generate. The default value is set to *120*. 
      - *max_length*:
        - The max length of the sentence to feed to the model. The default value is set to *256*. 
  - convert_to_active():
    - Converts a sentence to active voice, if found in passive voice. Otherwise returns the same sentence.
    - Parameters:
      - *sent*:
        - The sentence that has to be converted.
  - magic_once():
    - This is a wrapper method for *augment_sent_mask_filling()*, *augment_sent_t5()* and *convert_to_active()* methods. Using this, a sentence can be augmented using all the above mentioned techniques. 
    - Since this method can operate on individual text data, it can be merged with other packages.
    - Parameters:
      - *sent*:
        - The sentence that has to be augmented.
      - *paraphrase_prefix*:
        - The prefix for the T5 model input.
      - *n_paraphrase_predictions*:
        - The number of number augmentations, the function should return. The default value is set to *5*.
      - *paraphrase_top_k*:
        - The number of predictions, the T5 model should generate. The default value is set to *120*. 
      - *paraphrase_max_length*:
        - The max length of the sentence to feed to the model. The default value is set to *256*. 
      - *n_mask_predictions*:
        - The number of predictions, the BERT mask filling model should generate. The default value is set to *None*.
      - *convert_to_active*:
        - If the sentence should be converted to active voice. The default value is set to *True*.
  - magic_lamp():
    - This method can be used for augmenting whole dataset. Currently accepted dataset formats are: `txt`,`csv`,`tsv` and `list`. 
    - If the dataset is in `list` or `txt` format, a list of augmented sentences will be returned. Also, a `txt` file with the name *sentences_aug.txt* is saved containing the output of the augmented data. 
    - If a dataset is in `csv` or `tsv` format with labels, the dataset will be augmented along with keeping in memory the labels for the new samples and a pandas dataframe of the augmented data will be returned. A `tsv` file will be generated with the augmented output with name `original_file_name_aug.tsv` 
    - Parameters:
      - *sentences*:
        - The dataset that has to be augmented. This can be a `Python List`, a `txt`, `csv` or `tsv` file.
      - *paraphrase_prefix*:
        - The prefix for the T5 model input.
      - *n_paraphrase_predictions*:
        - The number of number augmentations, the function should return. The default value is set to *5*.
      - *paraphrase_top_k*:
        - The number of predictions, the T5 model should generate. The default value is set to *120*. 
      - *paraphrase_max_length*:
        - The max length of the sentence to feed to the model. The default value is set to *256*. 
      - *n_mask_predictions*:
        - The number of predictions, the BERT mask filling model should generate. The default value is set to *None*.
      - *convert_to_active*:
        - If the sentence should be converted to active voice. The default value is set to *True*.
      - *label_column*:
        - The name of the column that contains labeled data. The default value is set to *None*. This parameter is not required to be set if the dataset is in a `Python List` or a `txt` file.
      - *data_column*:
        - The name of the column that contains data. The default value is set to *None*. This parameter too is not required if the dataset is a `Python List` or a `txt` file.
      - *column_names*:
        - If the `csv` or `tsv` does not have column names, a Python list has to be passed to give the columns a name. Since this function also accepts `Python List` and a `txt` file, the default value is set to *None*. But, if `csv` or `tsv` files are used, this parameter has to be set.
<!--te-->

## References
[Passive To Active](https://github.com/DanManN/pass2act)

## Links
Please find an in depth explanation about the library [on my blog](https://towardsdatascience.com/textgenie-augmenting-your-text-dataset-with-just-2-lines-of-code-23ce883a0715).

## License
Please check `LICENSE` for more details.

