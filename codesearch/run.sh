lang=java #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
idx=0 #test batch idx

python run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_query1.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models_seed1_query1/$lang  \
--model_name_or_path $pretrained_model \
--seed 1 && 
python run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_query1.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models_seed3_query1/$lang  \
--model_name_or_path $pretrained_model \
--seed 3 &&
python run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_query1.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models_seed5_query1/$lang  \
--model_name_or_path $pretrained_model \
--seed 5 &&
python run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_query1.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models_seed7_query1/$lang  \
--model_name_or_path $pretrained_model \
--seed 7 &&
python run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train_query1.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models_seed11_query1/$lang  \
--model_name_or_path $pretrained_model \
--seed 11


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
--seed 1 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models_seed1_query1/$lang/checkpoint-best/ \
--test_result_dir ./result_seed1_query1/$lang/${idx}_batch_result.txt  &&
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
--pred_model_dir ./models_seed3_query1/$lang/checkpoint-best/ \
--test_result_dir ./result_seed3_query1/$lang/${idx}_batch_result.txt &&
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
--pred_model_dir ./models_seed5_query1/$lang/checkpoint-best/ \
--test_result_dir ./result_seed5_query1/$lang/${idx}_batch_result.txt &&
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
--pred_model_dir ./models_seed7_query1/$lang/checkpoint-best/ \
--test_result_dir ./result_seed7_query1/$lang/${idx}_batch_result.txt &&
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
--pred_model_dir ./models_seed11_query1/$lang/checkpoint-best/ \
--test_result_dir ./result_seed11_query1/$lang/${idx}_batch_result.txt 

# # do var
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_var_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed1_var/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 1 && 
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_var_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed3_var/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 3 &&
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_var_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed5_var/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 5 &&
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_var_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed7_var/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 7 &&
# python run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file train_var_adv.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../data/codesearch/train_valid/$lang \
# --output_dir ./models_seed11_var/$lang  \
# --model_name_or_path $pretrained_model \
# --seed 11


# python run_classifier.py \
# --model_type roberta \
# --model_name_or_path microsoft/codebert-base \
# --task_name codesearch \
# --do_predict \
# --output_dir ../data/codesearch/test/$lang \
# --data_dir ../data/codesearch/test/$lang \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 128 \
# --per_gpu_eval_batch_size 128 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --seed 1 \
# --test_file batch_${idx}.txt \
# --pred_model_dir ./models_seed1_var/$lang/checkpoint-best/ \
# --test_result_dir ./result_seed1_var/$lang/${idx}_batch_result.txt  &&
# python run_classifier.py \
# --model_type roberta \
# --model_name_or_path microsoft/codebert-base \
# --task_name codesearch \
# --do_predict \
# --output_dir ../data/codesearch/test/$lang \
# --data_dir ../data/codesearch/test/$lang \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --test_file batch_${idx}.txt \
# --pred_model_dir ./models_seed3_var/$lang/checkpoint-best/ \
# --test_result_dir ./result_seed3_var/$lang/${idx}_batch_result.txt &&
# python run_classifier.py \
# --model_type roberta \
# --model_name_or_path microsoft/codebert-base \
# --task_name codesearch \
# --do_predict \
# --output_dir ../data/codesearch/test/$lang \
# --data_dir ../data/codesearch/test/$lang \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --test_file batch_${idx}.txt \
# --pred_model_dir ./models_seed5_var/$lang/checkpoint-best/ \
# --test_result_dir ./result_seed5_var/$lang/${idx}_batch_result.txt &&
# python run_classifier.py \
# --model_type roberta \
# --model_name_or_path microsoft/codebert-base \
# --task_name codesearch \
# --do_predict \
# --output_dir ../data/codesearch/test/$lang \
# --data_dir ../data/codesearch/test/$lang \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --test_file batch_${idx}.txt \
# --pred_model_dir ./models_seed7_var/$lang/checkpoint-best/ \
# --test_result_dir ./result_seed7_var/$lang/${idx}_batch_result.txt &&
# python run_classifier.py \
# --model_type roberta \
# --model_name_or_path microsoft/codebert-base \
# --task_name codesearch \
# --do_predict \
# --output_dir ../data/codesearch/test/$lang \
# --data_dir ../data/codesearch/test/$lang \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --test_file batch_${idx}.txt \
# --pred_model_dir ./models_seed11_var/$lang/checkpoint-best/ \
# --test_result_dir ./result_seed11_var/$lang/${idx}_batch_result.txt 
