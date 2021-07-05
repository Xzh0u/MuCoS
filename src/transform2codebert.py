import pickle
import random


def main():
    data = pickle.load(open("data/train.all_augmented_data.pkl", 'rb'))

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
        if "permute_statement" in item.keys():
            adv_url.append(item['url'])
            adv_method_name.append(item['method_name'])
            adv_desc.append(item['docstring'].replace('\n', ''))
            adv_code.append(item["permute_statement"][0]['code'].replace(
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
    with open('train_permute_statement_full.txt', 'w') as f:
        for item in lines:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
