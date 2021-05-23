import torch
import argparse
import torch.nn.functional as F
from torch import nn
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from utils import processors
from run_classifier_new import evaluate
from transformers.modeling_bert import BertPreTrainedModel

MODEL_CLASSES = {'roberta': (
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}


class Ensemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        x1 = self.modelA(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, labels=labels)
        x2 = self.modelB(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, labels=labels)
        # x = torch.cat((x1[0], x2[0]), dim=1)
        l1 = self.softmax(x1)
        l2 = self.softmax(x2)
        x = (l1 + l2) / 2
        return x

class Ensemble3(BertPreTrainedModel):
    def __init__(self, modelA, modelB, modelC, config):
        super(Ensemble3, self).__init__(config)
        self.sub_modules = nn.ModuleDict({
            "modelA": modelA,
            "modelB": modelB,
            "modelC": modelC,
        })
        self.classifier = RobertaClassificationHead3(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        x1 = self.sub_modules["modelA"](input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
        x2 = self.sub_modules["modelB"](input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
        x3 = self.sub_modules["modelC"](input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
        x = torch.cat((x1[0], x2[0], x3[0]), dim=2)
        logits = self.classifier(x)
        return logits

class RobertaClassificationHead3(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size * 3, config.hidden_size * 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size * 3, config.num_labels)
        torch.nn.init.xavier_uniform(self.dense.weight)
        torch.nn.init.xavier_uniform(self.out_proj.weight)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="../data/codesearch/train_valid/java", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default="roberta", type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default='codesearch', type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default="./models/java", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=8, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port', type=str,
                        default='', help="For distant debugging.")
    parser.add_argument("--train_file", default="train.txt", type=str,
                        help="train file")
    parser.add_argument("--dev_file", default="valid.txt", type=str,
                        help="dev file")
    parser.add_argument("--test_file", default="batch_0.txt", type=str,
                        help="test file")
    parser.add_argument("--pred_modelA_dir", default=None, type=str,
                        help='./codesearch/models/java/checkpoint-best')
    parser.add_argument("--pred_modelB_dir", default=None, type=str,
                        help='./codesearch/models/java/checkpoint-best')
    parser.add_argument("--pred_modelX_dir", default=None, type=str,
                        help='./codesearch/models/java/checkpoint-best')
    parser.add_argument("--test_result_dir", default='./results/java/0_batch_result.txt', type=str,
                        help='path to store test result')
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.output_mode = "classification"

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)

    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
    elif args.model_name_or_path:
        tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name, do_lower_case=args.do_lower_case)

    modelX = model_class.from_pretrained(args.pred_modelX_dir)
    modelX.classifier = nn.Sequential(*list(modelX.classifier.children())[:-3])

    model1 = Ensemble3(modelX, modelX, modelX, config)

    modelA = model_class.from_pretrained(args.pred_modelA_dir, model_copy=model1, ensemble=True)
    modelB = model_class.from_pretrained(args.pred_modelB_dir, model_copy=model1, ensemble=True)
    

    model = Ensemble(modelA, modelB)
    model.to(device)
    evaluate(args, model, tokenizer,
             checkpoint=None, prefix='', mode='test')


if __name__ == "__main__":
    main()
