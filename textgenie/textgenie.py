import torch
from tqdm import tqdm
from .grammar_utils import pass2act, is_passive
from transformers import T5ForConditionalGeneration, T5Tokenizer
from string import punctuation
import os
import pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)


set_seed(42)


class TextGenie:
    def __init__(
        self,
        paraphrase_model_name,
        mask_model_name=None,
        spacy_model_name="en",
        device="cpu",
    ):
        tqdm.write("Loading Paraphrase Model..")
        self.paraphrase_model = T5ForConditionalGeneration.from_pretrained(
            paraphrase_model_name
        )
        self.paraphrase_tokenizer = T5Tokenizer.from_pretrained(paraphrase_model_name)
        self.paraphrase_model = self.paraphrase_model.to(device)
        self.device = device

        if mask_model_name:
            tqdm.write("Loading Mask Fill Model..")
            from transformers import pipeline
            from string import punctuation
            import spacy

            self.mask_augmenter = pipeline("fill-mask", model=mask_model_name)
            self.nlp = spacy.load(spacy_model_name)

    def extract_keywords(self, sentence):
        result = []
        pos_tag = ["PROPN", "NOUN", "ADJ"]
        consider_tags = ["NUM"]
        pos_tag = pos_tag + consider_tags

        doc = self.nlp(sentence)

        for token in doc:
            if (
                token.text in self.nlp.Defaults.stop_words or token.text in punctuation
            ) and token.pos_ not in consider_tags:
                continue
            if token.pos_ in pos_tag:
                result.append(token.text)
        return list(set(result))

    def augment_sent_mask_filling(self, sent, n_mask_predictions=5):
        keywords = self.extract_keywords(sent)
        augmented_sents = []
        for keyword in keywords:
            masked_sent = sent.replace(keyword, self.mask_augmenter.tokenizer.mask_token, 1)
            augmented_sents.extend(
                [
                    generated_sent["sequence"]
                    for generated_sent in self.mask_augmenter(
                        masked_sent, top_k=n_mask_predictions
                    )
                    if generated_sent["sequence"].lower() != sent.lower()
                ]
            )
        return augmented_sents

    def augment_sent_t5(self, sent, prefix, n_predictions=5, top_k=120, max_length=256):
        text = prefix + sent + " </s>"
        encoding = self.paraphrase_tokenizer.encode_plus(
            text, pad_to_max_length=True, return_tensors="pt"
        )
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding[
            "attention_mask"
        ].to(self.device)

        beam_outputs = self.paraphrase_model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            max_length=max_length,
            top_k=top_k,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=n_predictions,
        )

        final_outputs = []
        for beam_output in beam_outputs:
            generated_sent = self.paraphrase_tokenizer.decode(
                beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            if (
                generated_sent.lower() != sent.lower()
                and generated_sent not in final_outputs
            ):
                final_outputs.append(generated_sent)
        return final_outputs

    def convert_to_active(self, sent):
        if is_passive(sent):
            return pass2act(sent)
        else:
            return sent

    def magic_once(
        self,
        sent,
        paraphrase_prefix,
        n_paraphrase_predictions=5,
        paraphrase_top_k=120,
        paraphrase_max_length=256,
        n_mask_predictions=None,
        convert_to_active=True,
    ):
        sent = sent.strip()
        output = []
        output.append(sent)
        output += self.augment_sent_t5(
            sent,
            paraphrase_prefix,
            n_paraphrase_predictions,
            paraphrase_top_k,
            paraphrase_max_length,
        )
        if n_mask_predictions and isinstance(n_mask_predictions, int):
            output += self.augment_sent_mask_filling(sent, n_mask_predictions)
        if convert_to_active:
            active_voice = self.convert_to_active(sent)
            if active_voice.lower() != sent.lower():
                output.append(active_voice)
        return list(set(output))

    def magic_lamp(
        self,
        sentences,
        paraphrase_prefix,
        n_paraphrase_predictions=5,
        paraphrase_top_k=120,
        paraphrase_max_length=256,
        n_mask_predictions=None,
        convert_to_active=True,
        label_column=None,
        data_column=None,
        column_names=None,
    ):
        all_sentences = None
        with_labels = False
        out_file = os.path.join(os.getcwd(), "sentences_aug.txt")

        if isinstance(sentences, str):
            sentences = os.path.join(os.getcwd(), sentences)
            if sentences.endswith(".txt"):
                all_sentences = open(sentences).read().strip().split("\n")
            elif sentences.endswith(".csv") or sentences.endswith(".tsv"):
                if not label_column:
                    raise Exception(
                        "Please provide the column name that contains labels using the 'label_column' parameter."
                    )
                if not data_column:
                    raise Exception(
                        "Please provide the column name that contains data using the 'data_column' parameter."
                    )
                if column_names and not isinstance(column_names, list):
                    raise Exception("Please provide column names in a python list.")
                out_file = (
                    sentences.replace(".csv", "").replace(".tsv", "") + "_aug.tsv"
                )
                with_labels = True
                if sentences.endswith(".csv"):
                    if column_names:
                        all_sentences = pd.read_csv(sentences, names=column_names)
                    else:
                        all_sentences = pd.read_csv(sentences)
                elif sentences.endswith(".tsv"):
                    if column_names:
                        all_sentences = pd.read_csv(
                            sentences, names=column_names, sep="\t"
                        )
                    else:
                        all_sentences = pd.read_csv(sentences)
                if label_column not in all_sentences.columns:
                    raise Exception(
                        "Please provide label column name for the dataset using the 'label_column' parameter. If already provided, please check for typos in the name of the label column."
                    )
                if data_column not in all_sentences.columns:
                    raise Exception(
                        "Please provide data column name for the dataset using the 'data_column' parameter. If already provided, please check for typos in the name of the data column."
                    )
                labels = all_sentences[label_column].unique()

                if all_sentences.iloc[0].tolist() == column_names:
                    all_sentences = all_sentences.drop(0)
                augmented_data = []

                for ix in tqdm(range(all_sentences.shape[0])):
                    sent = all_sentences[data_column][ix].strip()
                    label = all_sentences[label_column][ix].strip()
                    aug_sent = self.magic_once(
                        sent,
                        paraphrase_prefix,
                        n_paraphrase_predictions,
                        paraphrase_top_k,
                        paraphrase_max_length,
                        n_mask_predictions,
                        convert_to_active,
                    )
                    aug_sent = [[s, label] for s in aug_sent]
                    augmented_data.extend(aug_sent)
                augmented_data = pd.DataFrame(
                    data=augmented_data, columns=["Text", "Label"]
                )
                augmented_data.to_csv(out_file, sep="\t", index=None)
            else:
                raise Exception(
                    "Unsupported file format. Currently, following formats are supported: list/csv/tsv"
                )
        elif isinstance(sentences, list):
            all_sentences = sentences
        if all_sentences is None:
            raise Exception("Error: No sentences found.")
        if not with_labels:
            augmented_data = []
            for sent in tqdm(all_sentences):
                augmented_data.extend(
                    self.magic_once(
                        sent,
                        paraphrase_prefix,
                        n_paraphrase_predictions,
                        paraphrase_top_k,
                        paraphrase_max_length,
                        n_mask_predictions,
                        convert_to_active,
                    )
                )
            with open(out_file, "w") as f:
                for line in augmented_data:
                    f.write(line + "\n")
        tqdm.write(f"\nCompleted writing output to {out_file}.")
        return augmented_data
