from difflib import SequenceMatcher
from urllib.parse import urlsplit


class Util:
    def __init__(self):
        pass

    @staticmethod
    def find_sentence_similarity_ratio(sentence1, sentence2):
        return SequenceMatcher(None, sentence1, sentence2).ratio()

    @staticmethod
    def find_domain_name(url):
        return "{0.scheme}://{0.netloc}/".format(urlsplit(url))