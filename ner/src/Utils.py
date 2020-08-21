#coding:utf-8

import datetime
import json
import os
from shutil import copyfile
import time
import re

import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

def meminfo(gpu_id):
    with tf.device('/device:GPU:{}'.format(gpu_id)):  # Replace with device you are interested in
      bytes_in_use = BytesInUse()
    with tf.Session() as sess:
      print('/device:GPU:{} mem usage:{}'.format(gpu_id, sess.run(bytes_in_use)))

def load_file(file):
    with open(file, 'r') as content_file:
        content = content_file.read()
    return content


def file_append(output_file, msg):
    with open(output_file, "a") as f:
        f.write(msg)


def file_write(output_file, msg):
    with open(output_file, "w") as f:
        f.write(msg)

def fwrite_append(output_file, text):
    with open(output_file, "a") as f:
        f.write(text)


def delete_file(file):
    try:
        os.remove(file)
    except Exception as e:
        print(e)


def mkdir(path):
    try:
        os.mkdir(path)
    except Exception as e:
        log(e)
        path


def log(msg):
    formatted_msg = "{} {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    file_append("log.txt", formatted_msg + "\n")
    print(formatted_msg)


def duplicate_file(folder, copy_cnt):
    file_list = []
    st = time.time()
    for dirpath, dirnames, files in os.walk(folder):
        for idx, file in enumerate(files):
            file_list.append(file)
    print("Total files:{}".format(len(file_list)))

    for file in file_list:
        for i in range(copy_cnt):
            copyfile(folder + "/" + file, folder + "/" + str(i) + "_" + file)
    et = time.time()
    print("finished:{}".format(et - st))


def remove_file(folder):
    file_list = []
    st = time.time()
    for dirpath, dirnames, files in os.walk(folder):
        for idx, file in enumerate(files):
            if file[0:2] == '0_' or file[0:2] == '1_' or file[0:2] == '2_' or file[0:2] == '3_':
                file_list.append(file)
    print("Total files:{}".format(len(file_list)))

    for file in file_list:
        delete_file(folder + "/" + file)
    et = time.time()
    print("finished:{}".format(et - st))


def list_folder(path):
    result = []
    for dirpath, dirnames, files in os.walk(path):
        for idx, file in enumerate(files):
            result.append(dirpath + "/" + file)
    return sorted(result)


def list_dir_in_path(path):
    result = []
    for dirpath, dirnames, files in os.walk(path):
        for idx, file in enumerate(files):
            if os.path.isdir(dirpath + "/" + file) == True:
                result.append(dirpath + "/" + file)
    return sorted(result)


def dump_json(file_path, data):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)


def load_json(file_path):
    with open(file_path, 'r') as content_file:
        content = content_file.read()
    return json.loads(content)

def is_whitespace(c):
    if c == " " or c == "　" or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def remove_symbols(s):
    return re.sub(r'[^\w]', '', s)


def remove_whitespace(text):
    ret_text = ""
    for c in text:
        if is_whitespace(c) is False:
            ret_text = ret_text + c
            
    return ret_text

def dump_lines(lines, output_file):
    for idx, line in enumerate(lines):
        file_append(output_file, '{}'.format(line))
    log('dump {} lines to {}'.format(len(lines), output_file))

def unique(values):
    result = []
    for value in values:
        if value in result:
            continue

        if len(value) == 2:
            if "女" in value or "男" in value:
                continue
        
        if len(value) == 2:
            should_pass = False
            
            #檢查2個字的名字是否出現在其他名字裡面
            for text in values:
                if value != text and value in text:
                    print(f"pass {value}")
                    should_pass = True
                    break
            if should_pass:
                continue

        result.append(value)
    return result