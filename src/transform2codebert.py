import pickle
import random
import argparse
'''use permute statement as en example'''


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # parser.add_argument("--data_dir", default="../data/train.adv_data.pkl", type=str, required=True,
    #                     help="The input data dir.")
    parser.add_argument("--code_type", default="origin", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_split", default="train", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    args = parser.parse_args()

    data = pickle.load(open(f"../data/{args.data_split}.adv_data.pkl", 'rb'))

    url = []
    method_name = []
    desc = []
    code = []
    adv_url = []
    adv_method_name = []
    adv_desc = []
    adv_code = []

    lines = []
    # first we get positive sample
    for item in data:
        url.append(item['url'])
        method_name.append(item['method_name'])
        desc.append(item['docstring'].replace('\n', ''))
        code.append(item['code'].replace('\n', '').replace(
            '\r', '').replace('    ', ''))
        if args.code_type == "origin":
            adv_url.append(item['url'])
            adv_method_name.append(item['method_name'])
            adv_desc.append(item['docstring'].replace('\n', ''))
            adv_code.append(item['code'].replace(
                '\n', '').replace('\r', '').replace('    ', ''))
        elif args.code_type in item.keys():
            if args.code_type == "replace_all":
                adv_url.append(item['url'])
                adv_method_name.append(item['method_name'])
                adv_desc.append(item['docstring'].replace('\n', ''))
                adv_code.append(item[args.code_type]['code'].replace(
                    '\n', '').replace('\r', '').replace('    ', ''))
            elif args.code_type == "permute_statement":
                adv_url.append(item['url'])
                adv_method_name.append(item['method_name'])
                adv_desc.append(item['docstring'].replace('\n', ''))
                adv_code.append(item[args.code_type][0]['code'].replace(
                    '\n', '').replace('\r', '').replace('    ', ''))

    print(len(adv_code))
    for i in range(len(code) - 2):
        line = "1<CODESPLIT>" + url[i] + "<CODESPLIT>" + method_name[i] + \
            "<CODESPLIT>" + desc[i] + "<CODESPLIT>" + code[i]
        lines.append(line)

    for i in range(len(adv_code) - 2):
        line = "1<CODESPLIT>" + adv_url[i] + "<CODESPLIT>" + adv_method_name[i] + \
            "<CODESPLIT>" + adv_desc[i] + "<CODESPLIT>" + adv_code[i]
        lines.append(line)

    for i in range(len(code) - 2):
        line = "0<CODESPLIT>" + url[i] + "<CODESPLIT>" + method_name[i] + \
            "<CODESPLIT>" + desc[i + 1] + "<CODESPLIT>" + code[i + 2]
        lines.append(line)

    for i in range(len(adv_code) - 2):
        line = "0<CODESPLIT>" + adv_url[i] + "<CODESPLIT>" + adv_method_name[i] + \
            "<CODESPLIT>" + adv_desc[i + 1] + "<CODESPLIT>" + adv_code[i + 2]
        lines.append(line)

    print(len(lines))
    random.shuffle(lines)
    with open(f'../data/{args.data_split}_{args.code_type}.txt', 'w') as f:
        for item in lines:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
