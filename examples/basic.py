# coding: utf-8

from textgenie import TextGenie

t5_model = "hetpandya/t5-base-tapaco"
bert_model = "microsoft/deberta-v3-large"

textgenie = TextGenie(t5_model, bert_model, spacy_model_name="en_core_web_lg", device="cuda")

# Augment a list of sentences
sentences = [
    "The video was posted on Facebook by Alex.",
    "I plan to run it again this time",
]

results = textgenie.magic_lamp(
    sentences, "paraphrase: ", n_mask_predictions=5, convert_to_active=True, add_suffix_token=False
)

print(results)
