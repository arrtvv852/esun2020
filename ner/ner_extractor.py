#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
import tokenization

import re
from difflib import SequenceMatcher
import datetime
import time

def list_dir_in_path(path):
    result = []
    for dirpath, dirnames, files in os.walk(path):
        for idx, file in enumerate(dirnames):
            if os.path.isdir(dirpath + "/" + file) == True:
                result.append(file)
    return sorted(result)

root_dir = "/app/ner"
# model_dir = f"{root_dir}/output-v2_0724/"
model_dir = f"{root_dir}/output-v1_0804/"
max_seq_len = 512
folders = list_dir_in_path("{}/export".format(model_dir))
saved_model_dir_name = folders[0]
print(saved_model_dir_name)

vocab_file = os.path.join(model_dir, "export/vocab.txt")
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

label_map_file = os.path.join(model_dir, "export/label_map.txt")
tags_map_file = os.path.join(model_dir, "export/tags_map.txt")
threshold_file = os.path.join(model_dir, "export/best_class_threshold.txt")

id2label = {}
with tf.gfile.GFile(label_map_file, "r") as reader:
    label_map_data = reader.read()
    label_map_data = label_map_data.split('\n')
    for data in label_map_data:
        try:
            id_, label = data.split(':')
            id2label[int(id_)] = label.strip('\'')
        except ValueError:  # lastest line is empty
            pass
num_labels = len(id2label)

# load tags map
id2tags = {}
with tf.gfile.GFile(tags_map_file, "r") as reader:
    tags_map_data = reader.read()
    tags_map_data = tags_map_data.split('\n')
    for data in tags_map_data:
        try:
            id_, tag = data.split(':')
            id2tags[int(id_)] = tag
        except ValueError:  # lastest line is empty
            pass
num_tags = len(id2tags)

# load threshold file
threshold_list = []
with tf.gfile.GFile(threshold_file, "r") as reader:
    threshold_data = reader.read()
    threshold_data = threshold_data.split('\n')
    for data in threshold_data:
        try:
            threshold_list.append(float(data))
        except ValueError:  # lastest line is empty
            pass

# load model set gpu_memory_fraction
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(graph=tf.Graph(), config=config)

saved_model_file = os.path.join(model_dir, "export/{}".format(saved_model_dir_name))
tf.saved_model.loader.load(sess, ['serve'], saved_model_file);


# In[121]:


def convert_single_example(text):

    tokens = []
    tokens.append("[CLS]")

    token = tokenizer.tokenize(text)
    tokens.extend(token)

    if len(tokens) >= max_seq_len:
        tokens = tokens[0: max_seq_len]

    segment_ids = [0] * max_seq_len
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    return input_ids, input_mask, segment_ids

def process_slot_output(slot_pred, input_ids, input_mask):
    slot_output = []
    for i in range(len(slot_pred)):
        slot_sentence = {}
        sentence_len = sum(input_mask[i])
        toknes = tokenizer.convert_ids_to_tokens(input_ids[i])
        # print(''.join(toknes[1:sentence_len]))
        slot_begin = -1
        slot_end = -1
        for j in range(sentence_len):  # 第i句的句子長度
            tag = slot_pred[i, j]
            tag = id2tags[tag]
            if tag == '[CLS]':
                slot = ''
                slot_type = ''
                slot_begin = -1
                slot_end = -1
                continue

            if tag.startswith('B'):
                slot_begin = j
                slot_end = j
                if slot == '':
                    slot_type = tag #tag.split('_')[1]
                    slot += toknes[j]
                else:  # slot不為空的話，代表遇到新的slot，先把先前的slot寫進output
                    if slot_type in slot_sentence.keys():
                        slot_sentence[slot_type] += [{"text":slot, "begin":slot_begin, "end": slot_end}]
                    else:
                        slot_sentence[slot_type] = [{"text":slot, "begin":slot_begin, "end": slot_end}]
                    slot = ''
                    slot_type = tag.split('_')[1]
                    slot += toknes[j]

            elif tag.startswith('I'):
                slot_end = j
                i_type = tag.split('_')[1]
                if i_type == slot_type:
                    slot += toknes[j]
                elif slot_type == '':
                    pass
                else:
                    if slot_type in slot_sentence.keys():
                        slot_sentence[slot_type] += [{"text":slot, "begin":slot_begin, "end": slot_end}]
                    else:
                        slot_sentence[slot_type] = [{"text":slot, "begin":slot_begin, "end": slot_end}]
                    slot_type = i_type
                    slot = toknes[j]
            slot = slot.replace('#', '')
        if slot != '':
            if slot_type in slot_sentence.keys():
                slot_sentence[slot_type] += [{"text":slot, "begin":slot_begin, "end": slot_end}]
            else:
                slot_sentence[slot_type] = [{"text":slot, "begin":slot_begin, "end": slot_end}]
        slot_output.append(slot_sentence)

    return slot_output

def process_intent_output(intent_pred):
    intent_output = []
    for i in range(len(intent_pred)):
        intent_sentence = []
        for j in range(num_labels):
            if intent_pred[i, j] >= threshold_list[j]:
                intent_sentence.append({'value': id2label[j], 'confidence': str(intent_pred[i, j])})
        intent_output.append(intent_sentence)
    return intent_output

def extract(article):
    input_ids, input_mask, segment_ids = convert_single_example(article)
    batch_input_ids = [input_ids]
    batch_input_mask = [input_mask]
    batch_segment_ids = [segment_ids]
    _, slot_pred = sess.run(['intent_loss/intent_probabilities:0', 'slot_loss/slot_predict:0'],
                                           feed_dict={'input_ids_1:0': batch_input_ids,
                                                      'input_mask_1:0': batch_input_mask,
                                                      'segment_ids_1:0': batch_segment_ids})

    slot_output = process_slot_output(slot_pred, batch_input_ids, batch_input_mask)
    print(slot_output)
    result = slot_output[0]
    if "NAME" in result:
        result["NAME"][0]["text"] = result["B_NAME"][0]["text"] + result["NAME"][0]["text"]
        result["NAME"][0]["begin"] = result["B_NAME"][0]["begin"]
        result["NAME"][0]["end"] = result["B_NAME"][0]["end"]
        del result["B_NAME"]
    else:
        result["NAME"] = ""
    
    fill_unk(article, result["NAME"])
    return distinct(result["NAME"])

def remove_symbols(s):
    return re.sub(r'[^\w]', '', s)

def similar(str1, str2):
    if str2 is None or str2.strip() == '':
        return 1.0
    return SequenceMatcher(None, remove_symbols(str1.lower()), remove_symbols(str2.lower())).ratio()

def value(obj, key):
    try:
        return obj[key][0]
    except:
        return ''
    
def distinct(slots):
    result = []
    for slot in slots:
        if slot["text"] in result:
            continue
        result.append(slot["text"])
    return result

def is_match(predict_names, names):
    if len(predict_names) != len(names):
        return False
    for name in predict_names:
        if name not in names:
            return False
    return True

def has_UNK(names):
    for name in names:
        if "[UNK]" in name:
            return True
    return False


def fill_unk(article, slots):
    for slot in slots:
        if "[UNK]" in slot["text"]:
            text = slot["text"].replace("[UNK]", "#")
            length = len(text)
            pos = text.index("#")
            
            if length > 4 and pos < (length - 1):
                text = text[0: pos + 1]
                length = len(text)
                pos = text.index("#")
            
            if length > 2 and pos < (length - 1):
                begin = article.index(text[0])
                end = article.index(text[pos + 1:], begin)
                slot["text"] = article[begin: end + 1]
            
            if length > 2 and pos == (length - 1):
                begin = article.index(text[0: pos])
                slot["text"] = article[begin: begin + length]
            
            if length == 2 and pos == 0:
                end = article.index(text[1])
                slot["text"] = article[end - 1: end + 1]
                
            if length == 2 and pos > 0:
                begin = article.index(text[0])
                slot["text"] = article[begin: begin + length]
            print("{} => {}".format(text, slot["text"]))

def validate(file):
    with open(file, 'r') as content_file:
        content = content_file.read()
    
    lines = content.split("\n")
    total = len(lines)
    print(total)
    cnt = 0
    for idx, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue
        tokens = line.split("\t")
        predict_names = extract(tokens[1])
        names = tokens[3].split(",")

        matched = is_match(predict_names, names)
        if matched:
            cnt = cnt + 1
        
        if has_UNK(predict_names):
            print(f"{matched} predict: {distinct_names} {names} \n")

    print("Acc: {}/{}={}".format(cnt, total, cnt/total))
    


print(f"NER Extractor [{model_dir}] is ready!!")
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# st = datetime.datetime.now()
# validate(f'{root_dir}/data-v1_0704/test.tsv')
# et = datetime.datetime.now()
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# print(et - st)
