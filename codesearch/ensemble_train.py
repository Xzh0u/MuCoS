import copy
import torch
import argparse
import logging
import glob
import os
import torch.nn.functional as F
from torch import nn
from transformers import (WEIGHTS_NAME, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from transformers.modeling_bert import BertPreTrainedModel
from utils import processors
from run_classifier_new import evaluate, train, load_and_cache_examples, set_seed

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class Ensemble(BertPreTrainedModel):
    def __init__(self, modelA, modelB, config):
        super(Ensemble, self).__init__(config)
        self.sub_modules = nn.ModuleDict({
            "modelA": modelA,
            "modelB": modelB
        })
        #self.modelA = modelA
        #self.modelB = modelB
        self.classifier = RobertaClassificationHead(config)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        x1 = self.sub_modules["modelA"](input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
        x2 = self.sub_modules["modelB"](input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
        x = torch.cat((x1[0], x2[0]), dim=2)
        logits = self.classifier(x)
        return logits


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size * 2, config.hidden_size * 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size * 2, config.num_labels)
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


def from_pretrained(cls, pretrained_model_name_or_path, model_copy=0, ensemble=False, *model_args, **kwargs):
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_tf = kwargs.pop("from_tf", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            *model_args,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            **kwargs,
        )
        print("A1, config:", config, " model_kwargs:", model_kwargs)
    else:
        model_kwargs = kwargs

    # Load model
    if pretrained_model_name_or_path is not None:
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                # Load from a TF 1.0 checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
            elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                # Load from a TF 2.0 checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    "Error no file named {} found in directory {} or `from_tf` set to False".format(
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME +
                            ".index"], pretrained_model_name_or_path
                    )
                )
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            archive_file = pretrained_model_name_or_path
        elif os.path.isfile(pretrained_model_name_or_path + ".index"):
            assert (
                from_tf
            ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                pretrained_model_name_or_path + ".index"
            )
            archive_file = pretrained_model_name_or_path + ".index"
        else:
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path, postfix=(
                    TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME)
            )

        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
            )
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                    archive_file)
            else:
                msg = (
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url to model weight files named one of {} but "
                    "couldn't find any such file at this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(cls.pretrained_model_archive_map.keys()),
                        archive_file,
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME],
                    )
                )
            raise EnvironmentError(msg)

        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))
    else:
        resolved_archive_file = None

    # Instantiate model.
    model = cls(config, *model_args, **model_kwargs)
    if ensemble:
        model = model_copy
    print("model!!:", model)
    if state_dict is None and not from_tf:
        try:
            state_dict = torch.load(
                resolved_archive_file, map_location="cpu")
        except Exception:
            raise OSError(
                "Unable to load weights from pytorch checkpoint file. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    if from_tf:
        if resolved_archive_file.endswith(".index"):
            # Load from a TensorFlow 1.X checkpoint - provided by original authors
            model = cls.load_tf_weights(
                model, config, resolved_archive_file[:-6])  # Remove the '.index'
        else:
            # Load from our TensorFlow 2.0 checkpoints
            try:
                from transformers import load_tf2_checkpoint_in_pytorch_model

                model = load_tf2_checkpoint_in_pytorch_model(
                    model, resolved_archive_file, allow_missing_keys=True)
            except ImportError:
                logger.error(
                    "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                    "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                )
                raise
    else:
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            model_to_load = getattr(model, cls.base_model_prefix)
        print(model_to_load)
        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

    model.tie_weights()  # make sure word embedding weights are still tied if needed

    # Set model in evaluation mode to desactivate DropOut modules by default
    model.eval()

    if output_loading_info:
        loading_info = {"missing_keys": missing_keys,
                        "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
        return model, loading_info

    return model


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
    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='predict model dir')
    parser.add_argument("--test_result_dir", default='./results/java/0_batch_result.txt', type=str,
                        help='path to store test result')
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(
            checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(
            checkpoint_last, args.start_epoch))

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
    elif args.model_name_or_path:
        tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name, do_lower_case=args.do_lower_case)

    modelA = model_class.from_pretrained(args.pred_modelA_dir)
    modelB = model_class.from_pretrained(args.pred_modelB_dir)

    # extract model without classifier
    modelA.classifier = nn.Sequential(*list(modelA.classifier.children())[:-3])
    modelB.classifier = nn.Sequential(*list(modelB.classifier.children())[:-3])

    model = Ensemble(modelA, modelB, config)
    model_copy = copy.deepcopy(model)
    # model.apply(init_weights)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, ttype='train')
        global_step, tr_loss = train(
            args, train_dataset, model, tokenizer, optimizer)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(
            args.output_dir, model_copy=model_copy, ensemble=True)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            print(checkpoint)
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(
                checkpoint, model_copy=model_copy, ensemble=True)
            model.to(args.device)
            result = evaluate(args, model, tokenizer,
                              checkpoint=checkpoint, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v)
                          for k, v in result.items())
            results.update(result)

    if args.do_predict:
        print('testing')
        model = model_class.from_pretrained(
            args.pred_model_dir, model_copy=model_copy, ensemble=True)
        model.to(args.device)
        evaluate(args, model, tokenizer,
                 checkpoint=None, prefix='', mode='test')
        return

    return results


if __name__ == "__main__":
    main()
