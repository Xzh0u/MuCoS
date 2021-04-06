# lang=java #fine-tuning a language-specific model for each programming language
# pretrained_model=microsoft/codebert-base  #Roberta: roberta-base

# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_struct_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 12 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed1/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 1 > seed1.txt && 
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_struct_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 12 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed3/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 3 > seed3.txt &&
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_struct_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 12 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed5/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 5 > seed5.txt &&
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_struct_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 12 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed7/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 7 > seed7.txt &&
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_struct_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 12 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed9/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 9 > seed9.txt 


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
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 12 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_seed1/$lang/checkpoint-best/ \
--test_result_dir ./result_seed1/$lang/${idx}_batch_result.txt &&
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
--num_train_epochs 12 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_seed3/$lang/checkpoint-best/ \
--test_result_dir ./result_seed3/$lang/${idx}_batch_result.txt &&
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
--num_train_epochs 12 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_seed5/$lang/checkpoint-best/ \
--test_result_dir ./result_seed5/$lang/${idx}_batch_result.txt &&
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
--num_train_epochs 12 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_seed7/$lang/checkpoint-best/ \
--test_result_dir ./result_seed7/$lang/${idx}_batch_result.txt &&
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
--num_train_epochs 12 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_seed9/$lang/checkpoint-best/ \
--test_result_dir ./result_seed9/$lang/${idx}_batch_result.txt 