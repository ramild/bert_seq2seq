# coding: utf-8
import re

import pymorphy2
import numpy as np
from keras import preprocessing

EMOTIPOS = " EMOTIPOS "
EMOTINEG = " EMOTINEG "
LICPLATE = " LICPLATE "
SCREAMER = " SCREAMER "
PHNUMBER = " PHNUMBER "
URL = " URL "
EMAIL = " EMAIL "
TERMINAL_WORDS = [
    EMOTINEG.strip(),
    EMOTIPOS.strip(),
    LICPLATE.strip(),
    SCREAMER.strip(),
    PHNUMBER.strip(),
    URL.strip(),
    EMAIL.strip(),
]

# List of substitution transformations to be applied to the texts.
REGEXP_TRANSFORMATIONS = [
    # Replaces emoticons with terminal words 'EMOTIPOS' or 'EMOTINEG'.
    (re.compile(r"(:D|;D|:-D|;-D|<3|:\*)"), EMOTIPOS),
    (re.compile(r"[:|;]?(\))\1+"), EMOTIPOS),
    (re.compile(r"[:|;]?(\()\1+"), EMOTINEG),
    # Replaces license plates with the terminal word 'LICPLATE'
    (re.compile(r"[^0-9\s]+[0-9]{3}[^0-9\s]+"), LICPLATE),
    # Replaces 3+ sequent exclamation marks with the terminal word 'SCREAMER'.
    (re.compile("!{3,}"), SCREAMER),
    # Replaces phone numbers with the terminal word 'PHNUMBER'.
    (re.compile(r"\+?[0-9]{5,12}"), PHNUMBER),
    # Replaces urls with the terminal word 'URL'.
    (re.compile(r"(https?://|www.)[^\s]*"), URL),
    # Replaces emails with the terminal word 'EMAIL'.
    (re.compile(r"[^\s]+@[^\s]+.[^\s]+"), EMAIL),
    # Cleans the text from invalid characters.
    (re.compile("[^A-Za-zА-Яа-яЁё0-9 ]"), " "),
    # Removes all words containing digits.
    (re.compile(r"[^\s]*[0-9][^\s]*"), ""),
    # Substitutes multiple whitespaces with a single whitespace.
    (re.compile(r"\s+"), " "),
    # Handles prolongations in the text: 'крууууто'->'крууто'.
    (re.compile(r"(.)\1+"), r"\1\1"),
]


class TextPreprocessor(object):
    """Class for preprocessing text data.

    This class allows cleaning the input text from unwanted characters,
    converting emoticons/license plates/multiple exclamation marks/phone
    numbers/urls/emails to terminal words, normalizing the words, and
    vectorizing the text.
    """

    MORPH = pymorphy2.MorphAnalyzer()

    @staticmethod
    def apply_regexes(text):
        """Apply regular expressions from TRANSFORMATIONS to the text."""
        text_clean = text
        for (pattern, repl) in REGEXP_TRANSFORMATIONS:
            text_clean = pattern.sub(repl, text_clean)
        return text_clean.strip()

    @staticmethod
    def _normalize_word(word):
        """Normalizes a word in Russian language."""
        if word in TERMINAL_WORDS:
            return word
        return TextPreprocessor.MORPH.parse(word)[0].normal_form

    @staticmethod
    def normalize_text(text):
        return " ".join(
            TextPreprocessor._normalize_word(word)
            for word in text.split()
            if len(word) > 1
        )

    @staticmethod
    def preprocess_text(text, encoding="utf-8"):
        if isinstance(text, str):
            return TextPreprocessor.normalize_text(
                TextPreprocessor.apply_regexes(text),
            )
        else:
            print("None str")
            return None

    @staticmethod
    def tokenize_text(
        text_preprocessed,
        tokenizer,
        max_num_words,
        encoding="utf-8",
    ):
        """Vectorizes the text by turning it into a sequence of integers
        each integer being the index of a token in a dictionary.

        Args:
            text_preprocessed (str or unicode): Preprocessed text.
            tokenizer (obj): Pretrained Keras tokenizer that converts an
                input text into a sequence of indexes of words in the
                dictionary. Considers only top-N most frequent words in
                the dictionary (based on tokenizer.word_counts).
            max_num_words (int): Upper bound for the number of words to be
                considered in a given text.
            encoding (str): encoding of the input text.

        Returns:
            Array-like keras.utils.Sequence() object representing a sequence
            of max_num_words integers.
        """
        sequence = tokenizer.texts_to_sequences([text_preprocessed])
        return preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=max_num_words,
        )

    @staticmethod
    def take_embeddings(text, vectors_embeddings):
        """Take pretrained w2vec or fasttext or glove words embeddings .

        Args:
            text (str or unicode): Preprocessed text.
            vectors_embeddings (obj): Pretrined words embeddings

        Returns:
            Array with zeros if there is no pretrained vector for this
            word and embedding vector if it exist.
        """
        embedding_array = []
        for word in text.split():
            if word in vectors_embeddings:
                embedding_array.append(vectors_embeddings[word])
            else:
                embedding_array.append(
                    np.zeros((1, vectors_embeddings.vector_size)),
                )

        return np.array(embedding_array).reshape(
            len(text.split()),
            vectors_embeddings.vector_size,
        )

    @staticmethod
    def preprocess_features(json_features, config):
        """Takes a json with features, processes categorical
        features, selects only necessary features
        in the specified order.

        Args:
            json_features - json with feature values
            config - config with information about values

        Returns:
            Array with feature values.
        """
        for feature in config["features_order"]:
            if feature not in json_features:
                json_features[feature] = config["empty_features"][feature]
        feature_list = []
        for feature in config["features_order"]:
            if feature in config["features_categ"]:
                for feature_value in config["features_categ"][feature]:
                    if (feature == "whats_wrong") and (
                        feature_value in json_features[feature].split(", ")
                    ):
                        feature_list.append(1)
                    elif feature_value == json_features[feature]:
                        feature_list.append(1)
                    else:
                        feature_list.append(0)
            elif feature in json_features:
                feature_list.append(float(json_features[feature]))
        return np.array(feature_list).reshape(1, len(feature_list))
