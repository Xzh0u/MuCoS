lang=java #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
idx=0

# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --seed 1 \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_origin-full_seed1/$lang  \
# --model_name_or_path $pretrained_model \

# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --seed 3 \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_origin-full_seed3/$lang  \
# --model_name_or_path $pretrained_model \

# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --seed 5 \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_origin-full_seed5/$lang  \
# --model_name_or_path $pretrained_model \

# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --seed 7 \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_origin-full_seed7/$lang  \
# --model_name_or_path $pretrained_model \

# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --seed 11 \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_origin-full_seed11/$lang  \
# --model_name_or_path $pretrained_model \

python run_classifier.py \
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
--test_file batch_${idx}.txt \
--pred_model_dir ./models_origin-full_seed1/$lang/checkpoint-best/ \
--test_result_dir ./results_seed1_origin-full/$lang/${idx}_batch_result.txt  &&
python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_origin-full_seed3/$lang/checkpoint-best/ \
--test_result_dir ./results_seed3_origin-full/$lang/${idx}_batch_result.txt &&
python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_origin-full_seed5/$lang/checkpoint-best/ \
--test_result_dir ./results_seed5_origin-full/$lang/${idx}_batch_result.txt &&
python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_origin-full_seed7/$lang/checkpoint-best/ \
--test_result_dir ./results_seed7_origin-full/$lang/${idx}_batch_result.txt &&
python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_origin-full_seed11/$lang/checkpoint-best/ \
--test_result_dir ./results_seed11_origin-full/$lang/${idx}_batch_result.txt 
