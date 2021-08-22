from typing import Dict, List
import os
import argparse
from utils import read_pickle, write_pickle
from parse_ast import extract_api, extract_method_name, extract_tokens
'''save the data in a pkl file
[{
    key: idx,
    method name: [],
    api: [],
    tokens: [],
    desc: [],
    ...
    "replace_all": {
        code: [],
        method name: [],
        api: [],
        tokens: [],
    }
    "replace_random": [{
        index: int
        code: [],
        method name: [],
        api: [],
        tokens: [],
    }, {...}]
}, {...}]
>>> len(list(data['test'].keys()))
25929
>>> len(list(data['train'].keys()))
430184
>>> len(list(data['val'].keys()))
14387
'''


def load_origin_data(data_dir, partition) -> List[Dict]:
    origin_code = []
    data = read_pickle(data_dir)

    for i in list(data[partition].keys()):
        origin_code.append({
            "key": i,
            "split_method_name": None,
            "split_api": None,
            "split_tokens": None,
            "docstring_tokens": data[partition][i]['docstring_tokens'],
            "method_name": data[partition][i]['func_name'],
            "code": data[partition][i]['original_string'],
            "docstring": data[partition][i]['docstring'],
            "path": data[partition][i]['path'],
            "repo": data[partition][i]['repo'],
            "url": data[partition][i]['url'],
            "sha": data[partition][i]['sha'],
            "language": "java",
            "partition": partition
        })

    print(f"Number of {partition} code snaps: ", len(origin_code))
    return origin_code


def parse_code_data(data_list) -> List[Dict]:
    for data_dict in data_list:
        standard_code_snap = "public class Example {" + data_dict["code"] + "}"
        data_dict["split_method_name"] = extract_method_name(
            standard_code_snap)
        data_dict["split_tokens"] = extract_tokens(standard_code_snap)
        data_dict["split_api"] = extract_api(standard_code_snap)

    return data_list


def collect_data_change_all_variable(dir_path) -> Dict:
    ''' return a Dict of idx: code'''

    augmented_data = {}
    file_index = []
    files = os.listdir(dir_path)

    # only use the data which change all variable
    for file in files:
        if file.endswith("_0.java"):
            with open(os.path.join(dir_path, file), 'r') as f:
                augmented_data.update({int(file.split("-")[0]): f.read()})
                file_index.append(int(file.split("-")[0]))

    for file in files:
        if int(file.split("-")[0]) not in file_index:
            with open(os.path.join(dir_path, file), 'r') as f:
                augmented_data.update({int(file.split("-")[0]): f.read()})
                file_index.append(int(file.split("-")[0]))

    print("Num of replace all variables code snap: ", len(augmented_data))
    return augmented_data


def collect_data_change_random_variable(dir_path) -> Dict:
    ''' return a Dict of idx: [code1, code2, code3].'''
    augmented_data = {}
    file_index = []
    files = os.listdir(dir_path)

    # only use the data which change all variable
    for file in files:
        if not file.endswith("_0.java") and not file.endswith("_1.java"):
            with open(os.path.join(dir_path, file), 'r') as f:
                index = int(file.split("-")[0])
                if index in augmented_data.keys() and augmented_data[index] is not None:
                    augmented_data[index].append(f.read())
                else:
                    file_index.append(index)
                    augmented_data.update(
                        {index: [f.read()]})

    for file in files:
        if file.endswith("_1.java") and index in file_index:  # one var
            with open(os.path.join(dir_path, file), 'r') as f:
                index = int(file.split("-")[0])
                if index in augmented_data.keys() and augmented_data[index] is not None:
                    augmented_data[index].append(f.read())

    print("Num of replace random variables code snap: ", len(augmented_data))
    return augmented_data


def collect_data_change_structure(dir_path) -> Dict:
    ''' return a Dict of idx: [code1, code2, code3]'''

    augmented_data = {}
    file_index = []
    files = os.listdir(dir_path)

    for file in files:
        with open(os.path.join(dir_path, file), 'r') as f:
            index = int(file.split("-")[0])
            if index in augmented_data.keys() and augmented_data[index] is not None:
                augmented_data[index].append(f.read())
            else:
                file_index.append(index)
                augmented_data.update(
                    {index: [f.read()]})

    print("Num of permute statement code snap: ", len(augmented_data))
    return augmented_data


def add_adv_data(origin_data, root_path="output"):
    dir_path = os.path.join(root_path, "VariableRenaming")
    # load and append data change all variable
    replace_all_data = collect_data_change_all_variable(dir_path)
    for data in origin_data:  # data key maybe not in replace all data
        if data["key"] in replace_all_data.keys():
            code = replace_all_data[data["key"]]
            data.update({"replace_all": {"code": code}})
    # load and append data change random variable
    replace_random_data = collect_data_change_random_variable(dir_path)
    for data in origin_data:
        if data["key"] in replace_random_data.keys() and replace_random_data[data["key"]] is not None:
            code_list = replace_random_data[data["key"]]
            for i, code in enumerate(code_list):
                if "replace_random" not in data.keys():
                    data.update(
                        {"replace_random": [{"index": i, "code": code}]})
                else:
                    data["replace_random"].append({"index": i, "code": code})

    return origin_data


def add_structural_data(origin_data, root_path="output"):
    dir_path = os.path.join(root_path, "PermuteStatement")
    structural_data = collect_data_change_structure(dir_path)
    for data in origin_data:
        if data["key"] in structural_data.keys() and structural_data[data["key"]] is not None:
            code_list = structural_data[data["key"]]
            for i, code in enumerate(code_list):
                if "permute_statement" not in data.keys():
                    data.update(
                        {"permute_statement": [{"index": i, "code": code}]})
                else:
                    data["permute_statement"].append(
                        {"index": i, "code": code})
    return origin_data


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="../data/csn.pkl", type=str,
                        help="The input data dir.")
    parser.add_argument("--data_split", default="train", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    args = parser.parse_args()
    # 1. load origin data
    data = load_origin_data(data_dir=args.data_dir, partition=args.data_split)
    # 2. parse origin code
    data = parse_code_data(data)
    # 3. add adverarial data
    data = add_adv_data(data)
    data = add_structural_data(data)
    # 4. save
    write_pickle(data, f"../data/{args.data_split}.adv_data.pkl")


if __name__ == "__main__":
    main()
