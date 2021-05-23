<<<<<<< HEAD
lang=java #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
idx=0 #test batch idx

python ensemble_simple+train.py \
=======
python ensemble2.py \
>>>>>>> a3da582c6a665320a0e5dc4bd2a32a1b85f63896
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_origin.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
<<<<<<< HEAD
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/java \
--output_dir ./models_ensemble3_notrain_train_seed1/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed1_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed1_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed1_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed1_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_notrain_train_seed1/java/0_batch_result.txt \
--seed 1

python ensemble_simple+train.py \
=======
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed1/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed1/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed1_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed1/java/0_batch_result.txt

python ensemble2.py \
>>>>>>> a3da582c6a665320a0e5dc4bd2a32a1b85f63896
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_origin.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
<<<<<<< HEAD
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/java \
--output_dir ./models_ensemble3_notrain_train_seed3/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed3_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed3_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed3_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed3_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_notrain_train_seed3/java/0_batch_result.txt \
--seed 3

python ensemble_simple+train.py \
=======
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed3/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed3/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed3_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed3/java/0_batch_result.txt

python ensemble2.py \
>>>>>>> a3da582c6a665320a0e5dc4bd2a32a1b85f63896
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_origin.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
<<<<<<< HEAD
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/java \
--output_dir ./models_ensemble3_notrain_train_seed5/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed5_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed5_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed5_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed5_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_notrain_train_seed1/java/0_batch_result.txt \
--seed 5

python ensemble_simple+train.py \
=======
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed5/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed5/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed5_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed5/java/0_batch_result.txt

python ensemble2.py \
>>>>>>> a3da582c6a665320a0e5dc4bd2a32a1b85f63896
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_origin.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
<<<<<<< HEAD
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/java \
--output_dir ./models_ensemble3_notrain_train_seed7/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed7_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed7_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed7_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed7_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_notrain_train_seed7/java/0_batch_result.txt \
--seed 7

python ensemble_simple+train.py \
=======
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed7/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed7/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed7_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed7/java/0_batch_result.txt

python ensemble2.py \
>>>>>>> a3da582c6a665320a0e5dc4bd2a32a1b85f63896
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_origin.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
<<<<<<< HEAD
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/java \
--output_dir ./models_ensemble3_notrain_train_seed11/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed11_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed11_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed11_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed11_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_notrain_train_seed11/java/0_batch_result.txt \
--seed 11


# predict
python ensemble_simple+train.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 128 \
--per_gpu_eval_batch_size 128 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--pred_modelA_dir ./models_seed1_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed1_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed1_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed1_var_full/java/checkpoint-best/ \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_ensemble3_notrain_train_seed1/$lang/checkpoint-best/ \
--test_result_dir ./results_ensemble3_notrain_train_seed1/$lang/${idx}_batch_result.txt

python ensemble_simple+train.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 128 \
--per_gpu_eval_batch_size 128 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--pred_modelA_dir ./models_seed3_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed3_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed3_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed3_var_full/java/checkpoint-best/ \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_ensemble3_notrain_train_seed3/$lang/checkpoint-best/ \
--test_result_dir ./results_ensemble3_notrain_train_seed3/$lang/${idx}_batch_result.txt

python ensemble_simple+train.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 128 \
--per_gpu_eval_batch_size 128 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--pred_modelA_dir ./models_seed5_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed5_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed5_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed5_var_full/java/checkpoint-best/ \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_ensemble3_notrain_train_seed5/$lang/checkpoint-best/ \
--test_result_dir ./results_ensemble3_notrain_train_seed5/$lang/${idx}_batch_result.txt

python ensemble_simple+train.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 128 \
--per_gpu_eval_batch_size 128 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--pred_modelA_dir ./models_seed7_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed7_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed7_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed7_var_full/java/checkpoint-best/ \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_ensemble3_notrain_train_seed7/$lang/checkpoint-best/ \
--test_result_dir ./results_ensemble3_notrain_train_seed7/$lang/${idx}_batch_result.txt

python ensemble_simple+train.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 128 \
--per_gpu_eval_batch_size 128 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--pred_modelA_dir ./models_seed11_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed11_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed11_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed11_var_full/java/checkpoint-best/ \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_ensemble3_notrain_train_seed11/$lang/checkpoint-best/ \
--test_result_dir ./results_ensemble3_notrain_train_seed11/$lang/${idx}_batch_result.txt

=======
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed11/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed11/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed11_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed11/java/0_batch_result.txt
>>>>>>> a3da582c6a665320a0e5dc4bd2a32a1b85f63896
