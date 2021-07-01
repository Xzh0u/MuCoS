# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
from more_itertools import chunked
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--result_dir', type=str,
                        default="./results_seed1/java")
    parser.add_argument('--pic_name', type=str,
                        default="ensemble")
    args = parser.parse_args()
    # languages = ['ruby', 'go', 'php', 'python', 'java', 'javascript']
    languages = ['java']
    for language in languages:
        file_dir = args.result_dir
        ranks = []
        num_batch = 0
        for file in sorted(os.listdir(file_dir)):
            print(os.path.join(file_dir, file))
            with open(os.path.join(file_dir, file), encoding='utf-8') as f:
                batched_data = chunked(f.readlines(), args.test_batch_size)
                for batch_idx, batch_data in enumerate(batched_data):
                    num_batch += 1
                    correct_score = float(
                        batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
                    scores = np.array(
                        [float(data.strip().split('<CODESPLIT>')[-1]) for data in batch_data])
                    rank = np.sum(scores >= correct_score)
                    # if rank == 1:
                    #     print(batch_data[batch_idx].strip().split('<CODESPLIT>')[3])
                    ranks.append(rank)

        valid_ranks10 = [number for number in ranks if number <= 10]
        valid_ranks5 = [number for number in ranks if number <= 5]
        valid_ranks1 = [number for number in ranks if number <= 1]
        mean_mrr = np.mean(1.0 / np.array(ranks))  # ignoring the NaN value
        mean_frank = np.mean(np.array(valid_ranks10))
        std_frank = np.std(np.array(valid_ranks10))
        print("{} mrr: {}".format(language, mean_mrr))
        print("{} SuccessRate@10: {}".format(language,
              len(valid_ranks10)/len(ranks)))
        print("{} SuccessRate@5: {}".format(language, len(valid_ranks5)/len(ranks)))
        print("{} SuccessRate@1: {}".format(language, len(valid_ranks1)/len(ranks)))
        print("{} frank avg: {}".format(language, mean_frank))
        print("{} ranks std: {}".format(language, std_frank))
        # fig = plt.hist(np.array(valid_ranks10))
        # plt.savefig('pic/{}_distribution.png'.format(args.pic_name))
        # MRR_dict[language] = mean_mrr
    # for key, val in MRR_dict.items():
    #     print("{} mrr: {}".format(key, val))


if __name__ == "__main__":
    main()
