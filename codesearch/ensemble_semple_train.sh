python ensemble_2+3.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed1_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed1_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed1_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed1_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_full_seed1/java/0_batch_result.txt

python ensemble_2+3.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed3_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed3_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed3_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed3_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_full_seed3/java/0_batch_result.txt

python ensemble_2+3.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed5_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed5_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed5_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed5_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_full_seed5/java/0_batch_result.txt

python ensemble_2+3.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed7_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed7_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed7_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed7_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_full_seed7/java/0_batch_result.txt

python ensemble_2+3.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed11_api/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed11_noapi/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed11_struct_full/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed11_var_full/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble3_full_seed11/java/0_batch_result.txt

# python ensemble.py \
# --model_type roberta \
# --task_name codesearch \
# --do_predict \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --data_dir ../data/codesearch/test/java \
# --output_dir ../data/codesearch/test/java  \
# --model_name_or_path microsoft/codebert-base \
# --pred_modelA_dir ./models_seed1_switch2if/java/checkpoint-best/ \
# --pred_modelB_dir ./models_seed1_loop/java/checkpoint-best/ \
# --test_file batch_0.txt \
# --test_result_dir ./results_ensemble1-2_seed1/java/0_batch_result.txt

# python ensemble.py \
# --model_type roberta \
# --task_name codesearch \
# --do_predict \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --data_dir ../data/codesearch/test/java \
# --output_dir ../data/codesearch/test/java  \
# --model_name_or_path microsoft/codebert-base \
# --pred_modelA_dir ./models_seed3_switch2if/java/checkpoint-best/ \
# --pred_modelB_dir ./models_seed3_loop/java/checkpoint-best/ \
# --test_file batch_0.txt \
# --test_result_dir ./results_ensemble1-2_seed3/java/0_batch_result.txt

# python ensemble.py \
# --model_type roberta \
# --task_name codesearch \
# --do_predict \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --data_dir ../data/codesearch/test/java \
# --output_dir ../data/codesearch/test/java  \
# --model_name_or_path microsoft/codebert-base \
# --pred_modelA_dir ./models_seed5_switch2if/java/checkpoint-best/ \
# --pred_modelB_dir ./models_seed5_loop/java/checkpoint-best/ \
# --test_file batch_0.txt \
# --test_result_dir ./results_ensemble1-2_seed5/java/0_batch_result.txt

# python ensemble.py \
# --model_type roberta \
# --task_name codesearch \
# --do_predict \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --data_dir ../data/codesearch/test/java \
# --output_dir ../data/codesearch/test/java  \
# --model_name_or_path microsoft/codebert-base \
# --pred_modelA_dir ./models_seed7_switch2if/java/checkpoint-best/ \
# --pred_modelB_dir ./models_seed7_loop/java/checkpoint-best/ \
# --test_file batch_0.txt \
# --test_result_dir ./results_ensemble1-2_seed7/java/0_batch_result.txt

# python ensemble.py \
# --model_type roberta \
# --task_name codesearch \
# --do_predict \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --data_dir ../data/codesearch/test/java \
# --output_dir ../data/codesearch/test/java  \
# --model_name_or_path microsoft/codebert-base \
# --pred_modelA_dir ./models_seed11_switch2if/java/checkpoint-best/ \
# --pred_modelB_dir ./models_seed11_loop/java/checkpoint-best/ \
# --test_file batch_0.txt \
# --test_result_dir ./results_ensemble1-2_seed11/java/0_batch_result.txt

