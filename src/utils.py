# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function
import pickle
from nltk import FreqDist
from typing import Dict, List
import json
import numpy as np
import re
import nltk

import csv
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

PAD_ID, SOS_ID, EOS_ID, UNK_ID = [0, 1, 2, 3]


def read_pickle(data_path):
    return pickle.load(open(data_path, 'rb'))


def write_pickle(data: List, data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


def read_tsv(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            lines.append(line)
        return lines


def camel_case_split(identifier):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def split_(identifier):  # still have some variable separated by _
    return identifier.split('_')


def remove_stop_and_key_words(valuables):
    with open('data/stop_words.txt', 'r') as file:
        stop_words = file.read().split('\n')

    with open('data/java_keywords.txt', 'r') as file:
        key_words = file.read().split('\n')

    remove_list = stop_words + key_words
    for should_removed in remove_list:
        if should_removed in valuables:
            valuables.remove(should_removed)
    return valuables


def remove_stop_and_key_words(valuables):
    with open('data/stop_words.txt', 'r') as file:
        stop_words = file.read().split('\n')

    with open('data/java_keywords.txt', 'r') as file:
        key_words = file.read().split('\n')

    remove_list = stop_words + key_words
    for should_removed in remove_list:
        if should_removed in valuables:
            valuables.remove(should_removed)

    return valuables


def remove_short_words(min_length, word_list):
    new_word_list = []
    for word in word_list:
        if len(word) >= min_length:
            new_word_list.append(word)

    return new_word_list


def build_vocab(content: List[List], vocab_type, length) -> Dict:
    # content = [['new', 'big', 'integer'], ['schedule', 'drain', 'buffers'], ['resolve'], ['decode', 'byte', 'array', 'as', 'purgeable'], ['set', 'offset', 'in', 'zoo', 'keeper'], ['invoke', 'method'], ['create'], ['set', 'cluster'], ['of'], ['try', 'create', 'query', 'cache'], [
    #     'print', 'new', 'line'], ['get', 'new', 'add', 'ons'], ['put', 'service', 'user', 'interface', 'metadata'], ['stream', 'in'], ['bind', 'view', 'helper'], ['detect'], ['iterator'], ['get', 'delegate', 'adapter'], ['send'], ['calculate', 'size']]
    all_counts = FreqDist([x for sublist in content for x in sublist])
    vocab = all_counts.most_common(length)

    vocab_list = []
    for item in vocab:
        vocab_list.append(item[0])

    special_tokens = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab_dict = dict(zip(vocab_list, [x for x in range(4, length)]))

    with open(f'save/vocab.{vocab_type}.json', 'w') as f:
        json.dump({**special_tokens, **vocab_dict}, f)

    return vocab_dict


def build_vocabs(data, datatype) -> Dict:
    method_name, tokens, api, desc = data

    name_vocab = build_vocab(method_name, f"name_{datatype}", 10000)
    tokens_vocab = build_vocab(tokens, f"tokens_{datatype}", 10000)
    api_vocab = build_vocab(api, f"apiseq_{datatype}", 10000)
    desc_vocab = build_vocab(desc, f"desc_{datatype}", 10000)

    return name_vocab, tokens_vocab, api_vocab, desc_vocab


try:
    nltk.word_tokenize("hello world")
except LookupError:
    nltk.download('punkt')


def sent2indexes(sentence, vocab, maxlen):
    '''sentence: a string or list of string
       return: a numpy array of word indices
    '''
    def convert_sent(sent, vocab, maxlen):
        idxes = np.zeros(maxlen, dtype=np.int64)
        idxes.fill(PAD_ID)
        tokens = nltk.word_tokenize(sent.strip())
        idx_len = min(len(tokens), maxlen)
        for i in range(idx_len):
            idxes[i] = vocab.get(tokens[i], UNK_ID)
        return idxes, idx_len
    if type(sentence) is list:
        inds, lens = [], []
        for sent in sentence:
            idxes, idx_len = convert_sent(sent, vocab, maxlen)
            #idxes, idx_len = np.expand_dims(idxes, 0), np.array([idx_len])
            inds.append(idxes)
            lens.append(idx_len)
        return np.vstack(inds), np.vstack(lens)
    else:
        inds, lens = sent2indexes([sentence], vocab, maxlen)
        return inds[0], lens[0]


def indexes2sent(indexes, vocab, ignore_tok=PAD_ID):
    '''indexes: numpy array'''
    def revert_sent(indexes, ivocab, ignore_tok=PAD_ID):
        indexes = filter(lambda i: i != ignore_tok, indexes)
        toks, length = [], 0
        for idx in indexes:
            toks.append(ivocab.get(idx, '<unk>'))
            length += 1
            if idx == EOS_ID:
                break
        return ' '.join(toks), length

    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim == 1:  # one sentence
        return revert_sent(indexes, ivocab, ignore_tok)
    else:  # dim>1
        sentences, lens = [], []  # a batch of sentences
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines  # lines is a two dimension array


class CodesearchProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]  # nl
            text_b = line[4]  # code
            if (set_type == 'test'):
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)[:50]

        tokens_b = None
        if example.text_b:  # text_a -> NL, text_b -> code
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "codesearch":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "codesearch": CodesearchProcessor,
}

output_modes = {
    "codesearch": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "codesearch": 2,
}
