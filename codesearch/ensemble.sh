python ensemble.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 12 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed1_var/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed1_struct/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed1_switch2if/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed1_loop/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble1-4_seed1/java/0_batch_result.txt

python ensemble.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 12 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed3_var/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed3_struct/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed3_switch2if/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed3_loop/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble1-4_seed3/java/0_batch_result.txt

python ensemble.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 12 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed5_var/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed5_struct/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed5_switch2if/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed5_loop/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble1-4_seed5/java/0_batch_result.txt

python ensemble.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 12 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed7_var/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed7_struct/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed7_switch2if/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed7_loop/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble1-4_seed7/java/0_batch_result.txt

python ensemble.py \
--model_type roberta \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 12 \
--data_dir ../data/codesearch/test/java \
--output_dir ../data/codesearch/test/java  \
--model_name_or_path microsoft/codebert-base \
--pred_modelA_dir ./models_seed11_var/java/checkpoint-best/ \
--pred_modelB_dir ./models_seed11_struct/java/checkpoint-best/ \
--pred_modelC_dir ./models_seed11_switch2if/java/checkpoint-best/ \
--pred_modelD_dir ./models_seed11_loop/java/checkpoint-best/ \
--test_file batch_0.txt \
--test_result_dir ./results_ensemble1-4_seed11/java/0_batch_result.txt

# python ensemble.py \
# --model_type roberta \
# --task_name codesearch \
# --do_predict \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 12 \
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
# --num_train_epochs 12 \
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
# --num_train_epochs 12 \
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
# --num_train_epochs 12 \
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
# --num_train_epochs 12 \
# --data_dir ../data/codesearch/test/java \
# --output_dir ../data/codesearch/test/java  \
# --model_name_or_path microsoft/codebert-base \
# --pred_modelA_dir ./models_seed11_switch2if/java/checkpoint-best/ \
# --pred_modelB_dir ./models_seed11_loop/java/checkpoint-best/ \
# --test_file batch_0.txt \
# --test_result_dir ./results_ensemble1-2_seed11/java/0_batch_result.txt