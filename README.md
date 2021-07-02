# MuCoS

This repo provides the code for reproducing the experiments in MuCoS: Is a Single Model Enough? MuCoS: A Multi-Model EnsembleLearning for Semantic Code Search. MuCoS is a multi-model ensemble learning architecture for semantic code search.

### Dependency

- pip install torch==1.4.0
- pip install transformers==2.5.0
- pip install filelock more_itertools
- pip install tensorboardX scikit-learn matplotlib

## Code Search

### Data Generation

To be finished.

### Fine-Tune

```shell
cd TODO:

lang=java #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base

python ensemble_train.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--pred_modelA_dir ./models_seed7_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed7_var/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed7_struct/java/checkpoint-best/ \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--pred_model_dir ./models_ensemble3_sampled_train_seed7/$lang/checkpoint-best/ \ 
```

### Inference

```shell
lang=java #programming language
idx=0 #test batch idx

python ensemble_train.py \
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
--pred_modelA_dir ./models_seed11_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed11_var/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed11_struct/java/checkpoint-best/ \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_ensemble3_sampled_train_seed11/$lang/checkpoint-best/ \
--test_result_dir ./results_ensemble3_sampled_train_seed11/$lang/${idx}_batch_result.txt 
```

### Evaluation

```shell
python mrr.py
```

