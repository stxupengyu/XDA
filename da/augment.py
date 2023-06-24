import random
import copy
import re
import string
import nltk
import numpy as np

def nltk_line_tokenizer(line):
    return nltk.word_tokenize(line)

def mask_text(text, mask_ratio=0.5, cnt=0,
              substitute_verbalizers=['<extra_id_{}>'.format(i) for i in range(300)],
              allow_substitute_punctuation=False, at_least_one=False, unchanged_phrases=[], changed_word_list=[]):
    tokens = nltk_line_tokenizer(text)
    # print('text', text)
    # print('tokens', tokens)
    # assert 0
    n = len(tokens)
    unchanged_phrases = [x.lower() for x in unchanged_phrases]
    splited_unchanged_phrases = [nltk_line_tokenizer(x.lower()) for x in unchanged_phrases]
    changed_word_list = [x.lower() for x in changed_word_list]
    if allow_substitute_punctuation:
        candidate_idxs = np.ones(n)
        for i in range(n):
            for splited_unchanged_phrase in splited_unchanged_phrases:
                if ' '.join(tokens[i:i + len(splited_unchanged_phrase)]).lower() == ' '.join(
                        splited_unchanged_phrase):
                    candidate_idxs[i:i + len(splited_unchanged_phrase)] = 0
        candidate_idxs = [i for (i, x) in enumerate(candidate_idxs) if x == 1]
        # candidate_idxs=[i for i in range(n) if tokens[i].lower() not in unchanged_word_list]
        idxs_should_be_changed = [i for i in range(n) if tokens[i].lower() in changed_word_list]
        n = len(candidate_idxs)
        indices = sorted(list(set(random.sample(candidate_idxs, int(n * mask_ratio)) + idxs_should_be_changed)))
        # indices=sorted(random.sample(range(n),int(n*mask_ratio)))
    else:
        candidate_idxs = np.ones(n)
        for i in range(n):
            for splited_unchanged_phrase in splited_unchanged_phrases:
                if tokens[i] in string.punctuation:
                    candidate_idxs[i] = 0
                if ' '.join(tokens[i:i + len(splited_unchanged_phrase)]).lower() == ' '.join(
                        splited_unchanged_phrase):
                    candidate_idxs[i:i + len(splited_unchanged_phrase)] = 0
        candidate_idxs = [i for (i, x) in enumerate(candidate_idxs) if x == 1]
        # candidate_idxs=[i for i in range(n) if tokens[i] not in string.punctuation and tokens[i].lower() not in unchanged_word_list]
        idxs_should_be_changed = [i for i in range(n) if tokens[i].lower() in changed_word_list]
        n = len(candidate_idxs)
        indices = sorted(list(set(random.sample(candidate_idxs, int(n * mask_ratio)) + idxs_should_be_changed)))
    if at_least_one == True and len(indices) == 0:
        indices = sorted(random.sample(range(n), 1))
    masked_src, masked_tgt = "", []
    for i, idx in enumerate(indices):
        if i == 0 or idx != indices[i - 1] + 1:
            masked_tgt.append("")
        masked_tgt[-1] += " " + tokens[idx]
        tokens[idx] = "[MASK]"
    for i, token in enumerate(tokens):
        if i != 0 and token == "[MASK]" and tokens[i - 1] == "[MASK]":
            continue
        if token == "[MASK]":
            masked_src += " " + substitute_verbalizers[cnt]
            cnt += 1
        else:
            masked_src += " " + token
    return masked_src.strip(), masked_tgt


def predict_blanks(texts_to_be_augmented, tgt_texts, gen_blanks_func, args):

    aug_type = args.aug_type
    aug_kwargs = {'mask_ratio': args.mask_ratio, 'aug_type': args.aug_type,
                  'aug_kwargs': {'do_sample': args.do_sample, 'num_beams': args.num_beams, 'num_return_sequences': 1}}
    bad_words_ids = [[3], [19794], [22354]] + [[2163], [4273], [465], [150], [1525], [58]]
    aug_kwargs['bad_words_ids'] = bad_words_ids
    # cprint('texts_to_be_augmented', texts_to_be_augmented)
    # cprint('tgt_texts', tgt_texts)
    # print('def predict_blanks.aug_kwargs:{},aug_type:{}'.format(aug_kwargs, aug_type))
    if 'iter' in aug_type:
        batch_size = int(aug_type.split('_')[2])
        pred_blanks = []
        for (text_to_be_augmented, tgt_parts) in zip(texts_to_be_augmented, tgt_texts):
            blen = len(tgt_parts)
            new_tgt_parts = copy.deepcopy(tgt_parts)
            masked_idxs = list(range(blen))
            if aug_type.startswith('rand_iter'):
                random.shuffle(masked_idxs)
            text_parts = re.split('<extra_id_\d+>', text_to_be_augmented)
            for batch_idx in range(int(np.ceil(len(masked_idxs) / batch_size))):
                cnt = 0
                masked_id = masked_idxs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                masked_id = sorted(masked_id)
                new_text = ''
                for i in range(len(text_parts) - 1):
                    new_text += text_parts[i]
                    if i in masked_id:
                        new_text += '<extra_id_{}>'.format(cnt)
                        cnt += 1
                    else:
                        new_text += new_tgt_parts[i]
                new_text += text_parts[-1]
                # print('new_text', new_text)
                # print('aug_kwargs', aug_kwargs)
                # assert 0
                total_predictions, preds = gen_blanks_func([new_text], **aug_kwargs)
                # cprint('total_predictions', total_predictions)
                # cprint('preds', preds)
                # assert 0
                preds = preds[0][0]
                # print(new_text,preds)
                if len(preds) > len(masked_id):
                    preds = preds[:len(masked_id)]
                else:
                    for _ in range(len(masked_id) - len(preds)):
                        preds.append('')
                for (m_id, pred_blank) in zip(masked_id, preds):
                    # cprint('new_tgt_parts', new_tgt_parts)
                    # cprint('pred_blank', pred_blank)
                    # cprint('m_id', m_id)
                    new_tgt_parts[m_id] = pred_blank

            pred_blanks.append(new_tgt_parts)
    elif aug_type == 'default':
        total_predictions, pred_blanks = gen_blanks_func(texts_to_be_augmented, **aug_kwargs)
        pred_blanks = [pred_blank[0] for pred_blank in pred_blanks]
        # pred_blanks=pred_blanks[0]
    return pred_blanks


def recover_examples_from_blanks(pure_parts, pred_blanks, model_type='t5'):
    # example_lines=[['[MASK] x','[MASK] y'],['x [MASK] y', '[MASK] z']]
    # pred_blanks=[['a','b'],['c','d']]
    # return filled_parts=[['a x','b y'],['x c y','d z']]
    if model_type is None:
        lines = ' '.join([' '.join(x) for x in pure_parts])
        if '[MASK]' in lines:
            model_type = 'GLM'
        elif '<extra_id_0>' in lines:
            model_type = 't5'
    filled_parts = []
    for (parts, pred_blank) in zip(pure_parts, pred_blanks):
        current_blank = 0
        filled_parts.append([])
        for part in parts:
            output_tokens = []
            tokens = part.split()
            for token in tokens:
                if (model_type.lower() == 't5' and token.startswith('<extra_id_')) or (
                        model_type.lower == 'glm' and token.startswith('[MASK]')):
                    if current_blank < len(pred_blank):
                        output_tokens.append(pred_blank[current_blank])
                    current_blank += 1
                else:
                    output_tokens.append(token)
            filled_parts[-1].append(' '.join((' '.join(output_tokens)).split()).strip())
            # print('def recover_examples_from_blanks',filled_parts[-1])
    return filled_parts


def postprocess_texts(filled_parts):
    processed_parts = []
    for parts in filled_parts:
        processed_parts.append([])
        for part in parts:
            processed_parts[-1].append(part.strip(string.punctuation).strip())
    return processed_parts