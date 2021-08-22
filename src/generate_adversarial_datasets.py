import pickle
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="../data/csn.pkl", type=str, required=False,
                        help="origin data dir.")
    parser.add_argument("--output_dir", default="output/", type=str, required=False,
                        help="output dir of augmented data.")
    args = parser.parse_args()

    with open(args.data_dir, "rb") as f:
        data = pickle.load(f)

    # for new small csn, remove meta
    code_list = []
    for i in list(data['train'].keys()):
        code_list.append([str(i) + "-" + data['train'][i]['func_name'],
                          data['train'][i]['original_string']])
    print(len(code_list))

    for idx, item in enumerate(code_list):
        if len(item[1]) < 131072 and type(item[1]) == str:  # less than system's max args num
            subprocess.run(
                ["java", "-jar", "JavaMethodTransformer.jar", item[0], item[1], args.output_dir])


if __name__ == "__main__":
    main()
