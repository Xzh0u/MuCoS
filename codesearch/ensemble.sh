python ensemble2.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed1/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed1/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed1_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed1/java/0_batch_result.txt

python ensemble2.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed3/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed3/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed3_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed3/java/0_batch_result.txt

python ensemble2.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed5/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed5/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed5_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed5/java/0_batch_result.txt

python ensemble2.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed7/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed7/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed7_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed7/java/0_batch_result.txt

python ensemble2.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_ensemble3_full_seed11/java/checkpoint-best/ \
--pred_modelB_dir ./models_ensemble3_query_seed11/java/checkpoint-best/ \
--pred_modelX_dir ./models_seed11_origin/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble6_all_seed11/java/0_batch_result.txt