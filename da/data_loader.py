import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import logging
import random
import torch
import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, init_ids, input_ids, input_mask, masked_lm_labels):
        self.init_ids = init_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.masked_lm_labels = masked_lm_labels


def convert_examples_to_features(examples, max_seq_length, tokenizer, seed=12345):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    # ----
    # dupe_factor = 5
    masked_lm_prob = 0.15  #0.15
    max_predictions_per_seq = 20  #20
    rng = random.Random(seed)

    for (ex_index, example) in enumerate(examples):
        modified_example = example
        tokens_a = tokenizer.tokenize(modified_example)
        # Account for [CLS] and [SEP] "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        # take care of prepending the class label in this code
        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")
        masked_lm_labels = [-100] * max_seq_length

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            # making sure that masking of # prepended label is avoided
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)
        len_cand = len(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms_pos = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms_pos) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = tokens[cand_indexes[rng.randint(0, len_cand - 1)]]

            masked_lm_labels[index] = tokenizer.convert_tokens_to_ids([tokens[index]])[0]
            output_tokens[index] = masked_token
            masked_lms_pos.append(index)

        init_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            init_ids.append(0)
            input_ids.append(0)
            input_mask.append(0)

        assert len(init_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("init_ids: %s" % " ".join([str(x) for x in init_ids]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))

        features.append(
            InputFeatures(init_ids=init_ids,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          masked_lm_labels=masked_lm_labels))
    return features


def prepare_data(features):
    all_init_ids = torch.tensor([f.init_ids for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in features],
                                        dtype=torch.long)
    tensor_data = TensorDataset(all_init_ids, all_input_ids, all_input_mask, all_masked_lm_labels)
    return tensor_data

def read_txt(txt_path, args):
    f = open(txt_path, 'r')
    lines = []
    count = 0
    for line in f.readlines():
        line = line.strip()[:args.cut_length]
        count += 1
        if count > args.sample_num:
            break
        # process src and trg line
        lines.append(line)
    return lines

def get_train_examples(args):
    train_texts = read_txt(os.path.join(args.data_dir, args.train_texts), args)
    train_labels = read_txt(os.path.join(args.data_dir, args.train_labels), args)
    return train_texts, train_labels