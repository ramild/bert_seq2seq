import re
from nltk.corpus import stopwords as sw
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import chain


class ReutersPreprocessor:
    def __init__(
        self,
        stemmer=PorterStemmer(),
        stopwords=sw.words("english"),
        min_length=3,
    ):
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.min_length = min_length

    def tokenize(self, text):
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words if word not in self.stopwords]
        tokens = list(map(lambda token: PorterStemmer().stem(token), words))
        filtered_tokens = list(
            filter(
                lambda token: re.match("[a-zA-Z]+", token)
                and len(token) >= self.min_length,
                tokens,
            ),
        )
        return filtered_tokens

    def pre_process(self, documents):

        train_documents = documents[(documents.lewissplit == "TRAIN")]
        train_category_list = [doc for doc in train_documents["topics"]]
        train_category_set = set(chain(*train_category_list))
        test_documents = documents[(documents.lewissplit == "TEST")]

        # From the test set we need to remove the topics that are not present in the train set, if any
        test_documents.loc[:, "topics"] = test_documents.topics.apply(
            lambda x: [entry for entry in x if entry in train_category_set],
        )

        vectorizer = TfidfVectorizer(tokenizer=self.tokenize, min_df=2)
        vectorized_train_documents = vectorizer.fit_transform(
            train_documents["text"],
        )
        vectorized_test_documents = vectorizer.transform(
            test_documents["text"],
        )

        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform(
            [doc for doc in train_documents["topics"]],
        )
        test_labels = mlb.transform([doc for doc in test_documents["topics"]])

        return (
            vectorized_train_documents,
            train_labels,
            vectorized_test_documents,
            test_labels,
        )
