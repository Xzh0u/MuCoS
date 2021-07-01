# MuCoS

This repo provides the code for reproducing the experiments in MuCoS: Is a Single Model Enough? MuCoS: A Multi-Model EnsembleLearning for Semantic Code Search. MuCoS is a multi-model ensemble learning architecture for semantic code search.

### Dependency

- pip install torch==1.4.0
- pip install transformers==2.5.0
- pip install filelock more_itertools
- pip install tensorboard scikit-learn matplotlib

## Code Search

### Data Generation

To be finished.

### Fine-Tune

```shell
cd TODO:

lang=java #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base

python run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models/$lang  \
--model_name_or_path $pretrained_model \
```

### Inference

```shell
lang=java #programming language
idx=0 #test batch idx

python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models/$lang/checkpoint-best/ \
--test_result_dir ./results/$lang/${idx}_batch_result.txt
```

### Evaluation

```shell
python mrr.py
```

