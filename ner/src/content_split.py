
import json
import glob
import os

WORKING_PATH = os.getcwd()

DATASET_DIR_NAME = "dataset"
JSON_DIR_NEME = "labeled"
MODIFIED_DIR = "labeled_splited"


SRC_LABELED_JSON_PATH = os.path.join(WORKING_PATH, DATASET_DIR_NAME, JSON_DIR_NEME)
DST_MODIFIED_PATH = os.path.join(WORKING_PATH, DATASET_DIR_NAME, MODIFIED_DIR)

MAX_LEN_CONTENT = 512


# def split_content_xx( content, max_content_len ) :
#     splited_content_list = []
#     # print("ori len",len(content))
#     content_list = content.split(" ")
#     unit_concat_len = 0
#     unit_concat_list = []

#     for unit in content_list :

#         if unit_concat_len + len( unit ) + 1 < max_content_len :
#             unit_concat_list.append( unit )
#             unit_concat_len += len( unit ) + 1
#         else :
#             ori_len = len( unit_concat_list )
#             unit_concat_list = " ".join(unit_concat_list)
#             # print("    unit_concat_list len {}, {}".format(  len(unit_concat_list), ori_len) )
#             splited_content_list.append(unit_concat_list )
#             unit_concat_list = []
#             unit_concat_len = 0

#     if len(unit_concat_list) :
#         unit_concat_list = " ".join(unit_concat_list)
#         # print("    unit_concat_list len {} End".format( len(unit_concat_list) ) )
#         splited_content_list.append(unit_concat_list )

#     print(len(splited_content_list))
#     return splited_content_list


def split_content(content_list, max_content_len, split_str=" "):
    splited_content_list = []
    for content in content_list:
        content_len = len(content)
        start_idx = 0
        search_end_idx = 0
        search_len = max_content_len

        while start_idx + search_len < content_len:
            split_idx = -1
            for idx in range(start_idx, start_idx + search_len):
                if content[idx] == split_str:
                    split_idx = idx

            if split_idx != -1 and split_idx != start_idx:
                splited_content = content[start_idx: split_idx]
                splited_content_list.append(splited_content)
                start_idx = split_idx
                search_end_idx = split_idx
            else:
                start_idx = start_idx + search_len

        if search_end_idx < content_len:
            splited_content = content[search_end_idx: content_len]
            splited_content_list.append(splited_content)

    return splited_content_list


def split_json_content(src_labeled_dir, dst_modified_dir, max_content_len = MAX_LEN_CONTENT) :
    all_jsons = os.path.join(src_labeled_dir,"*json")

    for json_file in glob.glob( all_jsons ) :
        # print(json_file)

        with open( json_file , 'r', encoding='utf-8') as f_srd:

            json_src_list = json.load(f_srd)
            # print(len(json_src_list) )
            json_modified_list = []
            json_file_name = os.path.split(json_file)[-1]

            modified_json_path = os.path.join( dst_modified_dir, json_file_name)

            for json_info_src in json_src_list :
                # print("ooo ", len(json_info_src["content"] ))
                splited_content_list = split_content( [ json_info_src["content"] ] , max_content_len, split_str="。")
                # print("aaa ",len(splited_content_list) )
                splited_content_list = split_content( splited_content_list, max_content_len, split_str="！")
                # print("bbb ",len(splited_content_list))
                splited_content_list = split_content( splited_content_list, max_content_len, split_str=" ")
                # print("ccc ",len(splited_content_list))

                for splited_content in splited_content_list :
                    json_info = json_info_src.copy()
                    json_info["content"] = splited_content
                    json_modified_list.append( json_info )

            with open(modified_json_path, 'w', encoding='utf-8') as f_dst:
                json.dump(json_modified_list, f_dst, ensure_ascii=False, indent=4)

    return



def split_context():
    print("split ---- begin")
    if not os.path.isdir(DST_MODIFIED_PATH) :
        os.mkdir(DST_MODIFIED_PATH)

    split_json_content(src_labeled_dir = SRC_LABELED_JSON_PATH,  dst_modified_dir = DST_MODIFIED_PATH, max_content_len = MAX_LEN_CONTENT)
    print("split ---- end")


def split_content_test(content):
    max_content_len = 512
    content_list = split_content([content], max_content_len, split_str="。")
    content_list = split_content(content_list, max_content_len, split_str="！")
    content_list = split_content(content_list, max_content_len, split_str=" ")
    return content_list

def split_test():
    with open("./dataset/labeled/20190918-labeled-ptt.json", 'r', encoding='utf-8') as f_srd:
        posts = json.load(f_srd)
        print(len(posts))
        for post in posts:
            if len(post["content"]) > 512:
                print('content length:{}'.format(len(post["content"])))
                print(post["content"])
                list = split_content_test(post["content"])
                for splited_content in list:
                    print(len(splited_content))
                break


# split_test()