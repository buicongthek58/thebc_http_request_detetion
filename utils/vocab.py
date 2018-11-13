from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import json


class Vocabulary():

    def __init__(self):
        self.vocab_file = os.path.dirname(os.path.abspath(__file__)) + '/vocab.json'
        with open(self.vocab_file, 'r') as f:
            self.vocab = json.load(f)

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def string_to_int(self, text):
        """
        Converts a string into its character integer representation.
        """
        try:
            characters = list(text)
        except Exception as e:
            characters = ['<UNK>']

        characters.append('<EOS>')
        char_ids = [self.vocab.get(char, self.vocab['<UNK>'])
                    for char in characters]

        return char_ids

    def int_to_string(self, char_ids):
        """
        Decodes a list of integers into it's string representation.
        """
        characters = []
        for i in char_ids:
            characters.append(self.reverse_vocab[i].encode('utf-8'))

        return characters


# ---------------------------------------------------------------
# import re
#
# HTTP_RE = re.compile(r"ST@RT.+?INFO\s+(.+?)\s+END", re.MULTILINE | re.DOTALL)
#
#
# def http_re(data):
#     """
#     Extracts HTTP requests from raw data string in special logging format.
#
#     Logging format `ST@RT\n%(asctime)s %(levelname)-8s\n%(message)s\nEND`
#     where `message` is a required HTTP request bytes.
#     """
#     return HTTP_RE.findall(data)
#
#
# def get_requests_from_file(path):
#     """
#     Reads raw HTTP requests from given file.
#     """
#     with open(path, 'r') as f:
#         file_data = f.read()
#     requests = http_re(file_data)
#     return requests
#
# path_anomaly_data = "datasets\\vulnbank_anomaly.txt"
#
# import os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data = get_requests_from_file('%s\%s' % (BASE_DIR, path_anomaly_data))
#
# vocab = Vocabulary()
# print(vocab.string_to_int(data[0]))


