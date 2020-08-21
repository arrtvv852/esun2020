import os
import tensorflow as tf
import tokenization

import re
from difflib import SequenceMatcher
import datetime
import time
from src.Utils import *


model_dir = './output_0917-v2/export/'
max_seq_len = 512

vocab_file = os.path.join(model_dir, "vocab.txt")
tokenizer = tokenization.FullTokenizer(
vocab_file=vocab_file, do_lower_case=True)

label_map_file = os.path.join(model_dir, "label_map.txt")
tags_map_file = os.path.join(model_dir, "tags_map.txt")
threshold_file = os.path.join(model_dir, "best_class_threshold.txt")

# load label map
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
config.gpu_options.per_process_gpu_memory_fraction = 0.1
print("@@@@@@ 0.1 @@@@@@")
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
sess = tf.Session(graph=tf.Graph(), config=config)

meminfo(0)
tf.saved_model.loader.load(sess, ['serve'], model_dir);
meminfo(0)

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
        for j in range(sentence_len):  # 第i句的句子長度
            tag = slot_pred[i, j]
            tag = id2tags[tag]
            if tag == '[CLS]':
                slot = ''
                slot_type = ''
                continue
            if tag.startswith('B'):
                if slot == '':
                    slot_type = tag.split('_')[1]
                    slot += toknes[j]
                else:  # slot不為空的話，代表遇到新的slot，先把先前的slot寫進output
                    if slot_type in slot_sentence.keys():
                        slot_sentence[slot_type] += [slot]
                    else:
                        slot_sentence[slot_type] = [slot]
                    slot = ''
                    slot_type = tag.split('_')[1]
                    slot += toknes[j]

            elif tag.startswith('I'):
                i_type = tag.split('_')[1]
                if i_type == slot_type:
                    slot += toknes[j]
                elif slot_type == '':
                    pass
                else:
                    if slot_type in slot_sentence.keys():
                        slot_sentence[slot_type] += [slot]
                    else:
                        slot_sentence[slot_type] = [slot]
                    slot_type = i_type
                    slot = toknes[j]
            slot = slot.replace('#', '')
        if slot != '':
            if slot_type in slot_sentence.keys():
                slot_sentence[slot_type] += [slot]
            else:
                slot_sentence[slot_type] = [slot]
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

def extract(message):
    input_ids, input_mask, segment_ids = convert_single_example(message)
    batch_input_ids = [input_ids]
    batch_input_mask = [input_mask]
    batch_segment_ids = [segment_ids]
    _, slot_pred = sess.run(['intent_loss/intent_probabilities:0', 'slot_loss/slot_predict:0'],
                                           feed_dict={'input_ids_1:0': batch_input_ids,
                                                      'input_mask_1:0': batch_input_mask,
                                                      'segment_ids_1:0': batch_segment_ids})

    slot_output = process_slot_output(slot_pred, batch_input_ids, batch_input_mask)
    slotObj = slot_output[0]
    store = value(slotObj, "STORE")
    phone = value(slotObj, "PHONE")
    address = value(slotObj, "ADDRESS")
    time = value(slotObj, "TIME")
    return store, phone, address, time

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

    
def validate(file):
    with open(file, 'r') as content_file:
        content = content_file.read()
    
    lines = content.split("\n")
    print(len(lines))
    threshold = 0.8
    
    correct4_count = 0
    
    correct3_count = 0
    
    correct2_count = 0
    
    correct1_count = 0
    
    correct0_count = 0

    total = len(lines)
    for idx, line in enumerate(lines):
        
        tokens = line.split("\t")
        if idx < total and len(tokens) >= 7:

            slotObj = extract(tokens[1])
            score = 0
            store = value(slotObj, "STORE")    
            if similar(store, tokens[3]) >= threshold:
                score = score + 1
                
            phone = value(slotObj, "PHONE")
            if similar(phone, tokens[4]) >= threshold:
                score = score + 1
            
            address = value(slotObj, "ADDRESS")
            if similar(address, tokens[5]) >= threshold:
                score = score + 1

            time = value(slotObj, "TIME")
            if similar(time, tokens[6]) >= threshold:
                score = score + 1
                
            if score >= 4:
                correct4_count = correct4_count + 1

            if score >= 3:
                correct3_count = correct3_count + 1
                
            if score >= 2:
                correct2_count = correct2_count + 1
                
            if score >= 1:
                correct1_count = correct1_count + 1
                
            if score >= 0:
                correct0_count = correct0_count + 1

            if idx % 100 == 0:
                print(idx)
#             print('{} {} {} {}'.format(idx, tokens[0], score, tokens[3:7]))

    print('{} {} {} {} {}'.format(correct4_count,
                                     correct3_count,
                                     correct2_count,
                                     correct1_count,
                                     correct0_count))

# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# st = datetime.datetime.now()
# validate('./data_0917-v2/test.tsv')
# et = datetime.datetime.now()
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# print(et - st)