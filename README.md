# MuCoS

This repo provides the code for reproducing the experiments in MuCoS: Is a Single Model Enough? MuCoS: A Multi-Model EnsembleLearning for Semantic Code Search. MuCoS is a multi-model ensemble learning architecture for semantic code search.

### Dependency

- pip install torch==1.4.0
- pip install transformers==2.5.0
- pip install javalang nltk numpy h5py # data process
- pip install filelock more_itertools tensorboardX scikit-learn matplotlib # run MuCoS

## Code Search

### Data Generation
The example CodeSearchNet data for the process is `csn.pkl`(this is available in ./).
Then we generate adversarial data using `generate_adversarial_datasets.py`, this file call a jar package in https://github.com/mdrafiqulrabin/tnpa-framework#1-variable-renaming to process the origin data and do 6 types of program transformation, then save the data to `data/output/<transform_type>/<filename>`. 
We use `save_dataset.py` to read all augmented data and add to origin CodeSearchNet data, then parse the api, tokens and method name, and save all these data to `valid.adv_data.pkl` and `train.all_augmented_data.pkl`. 


### Step 1: generate data
```python
python generate_adversarial_datasets.py
```
The output will be the same as in output/ of this repo.

### Step 2: parse and save augmented and origin data 
all augmented data into origin data, and parse origin and augmented data, save to a new pickle file
```python
python save_dataset.py
```
test data is saved to `test_train.adv_data.pkl`.

### Step 3: run scripts to build different dataset for each baseline
- put `valid.adv_data.pkl`(in directory root) to `data/valid.adv_data.pkl`
- put `train.all_augmented_data.pkl` to `data/train.all_augmented_data.pkl`

```python
# YOU SHOULD CHANGE DATA PATH BEFORE RUNNING!!!
python transform2codebert.py

```

### Fine-Tune

```shell
cd src

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

