import os
import fnmatch
import itertools
from pandas import DataFrame
import re
from html.parser import HTMLParser


class ReutersSGMLParser(HTMLParser):
    def __init__(self, encoding="latin-1"):
        HTMLParser.__init__(self)
        self.encoding = encoding
        self._reset()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""
        self.lewissplit = ""
        self.topics_attribute = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def handle_starttag(self, tag, attributes):
        if tag == "reuters":
            attributes_dict = dict(attributes)
            self.lewissplit = attributes_dict["lewissplit"]
            self.topics_attribute = attributes_dict["topics"]
        elif tag == "title":
            self.in_title = 1
        elif tag == "body":
            self.in_body = 1
        elif tag == "topics":
            self.in_topics = 1
        elif tag == "d":
            self.in_topic_d = 1

    def handle_endtag(self, tag):
        if tag == "reuters":
            self.body = re.sub(r"\s+", r" ", self.body)
            self.docs.append(
                {
                    "title": self.title,
                    "body": self.body,
                    "topics": self.topics,
                    "lewissplit": self.lewissplit,
                    "topics_attribute": self.topics_attribute,
                },
            )
            self._reset()
        elif tag == "title":
            self.in_title = 0
        elif tag == "body":
            self.in_body = 0
        elif tag == "topics":
            self.in_topics = 0
        elif tag == "d":
            self.in_topic_d = 0
            self.topics.append(self.topic_d)
            self.topic_d = ""


class ReutersReader:
    def __init__(self, data_path, split="ModApte"):
        self.data_path = data_path
        self.split = split

    def fetch_documents_generator(self):
        for root, _dirnames, filenames in os.walk(self.data_path):
            for filename in fnmatch.filter(filenames, "*.sgm"):
                path = os.path.join(root, filename)
                parser = ReutersSGMLParser()
                for doc in parser.parse(open(path, "rb")):
                    yield doc

    def get_documents(self):
        doc_generator = self.fetch_documents_generator()

        if self.split == "ModLewis":
            data = [
                (
                    "{title}\n\n{body}".format(**doc),
                    doc["topics"],
                    doc["lewissplit"],
                )
                for doc in itertools.chain(doc_generator)
                if doc["lewissplit"] != "NOT-USED"
                and doc["topics_attribute"] != "BYPASS"
            ]
        else:
            data = [
                (
                    "{title}\n\n{body}".format(**doc),
                    doc["topics"],
                    doc["lewissplit"],
                )
                for doc in itertools.chain(doc_generator)
                if doc["lewissplit"] != "NOT-USED" and doc["topics_attribute"] == "YES"
            ]

        return DataFrame(data, columns=["text", "topics", "lewissplit"])
