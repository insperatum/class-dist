import random
import functools
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

PROMPT = """[INST]You are a helpful assistant that classifies messages into categories of how effective they are at persuading people.
The categories are quantiles of effectiveness.

The categories are:
- [0, 0.2): very ineffective
- [0.2, 0.4): somewhat ineffective
- [0.4, 0.6): neutral
- [0.6, 0.8): somewhat effective
- [0.8, 1]: very effective

[\INST]

"""

def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    # multi-label classification
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

# define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    # labels are of shape (batch_size, num_labels) with probability for each label from 0 to 1
    d['labels'] = torch.stack(d['labels'])
    return d


# create custom trainer class to be able to pass label weights and calculate mutilabel loss
class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        # logits are of shape (batch_size, num_labels)
        logits = outputs.get("logits")
        
        # compute custom loss
        loss = -(F.softmax(logits, dim=1) * labels).sum(dim=1).log().mean()
        return (loss, outputs) if return_outputs else loss


def prepare_data(csv_path):
    with open(csv_path, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        header_row = data.pop(0)

    # shuffle data
    random.shuffle(data)

    # reshape
    text, labels = list(zip(*[(PROMPT + row[0], row[1:]) for row in data]))
    labels = np.array(labels)
    return text, labels

if __name__ == '__main__':
    # set random seed
    random.seed(0)

    # load train data
    train_text, train_labels = prepare_data('train.csv')
    val_text, val_labels = prepare_data('val.csv')

    # create hf dataset
    ds = DatasetDict({
        'train': Dataset.from_dict({'text': train_text, 'labels': train_labels}),
        'val': Dataset.from_dict({'text': val_text, 'labels': val_labels})
    })

    # model name
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

    # preprocess dataset with tokenizer
    def tokenize_examples(examples, tokenizer):
        tokenized_inputs = tokenizer(examples['text'])
        tokenized_inputs['labels'] = examples['labels']
        return tokenized_inputs

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=train_labels.shape[1],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # define training args
    training_args = TrainingArguments(
        output_dir = 'multilabel_llama-3.1-8b',
        learning_rate = 1e-6,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        warmup_steps = 100,
        num_train_epochs = 3,
        weight_decay = 0.01,
        evaluation_strategy = 'steps',
        save_strategy = 'steps',
        save_steps = 100,
        save_total_limit = 3,
        load_best_model_at_end = True,
        metric_for_best_model = 'valid_loss"',
        greater_is_better = False,
        logging_dir = 'logs',
        logging_strategy = 'steps',
        logging_steps = 10,
        report_to = 'none',
        bf16 = True,
        gradient_checkpointing = True,
        torch_compile = True
    )

    # train
    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_ds['train'],
        eval_dataset = tokenized_ds['val'],
        tokenizer = tokenizer,
        data_collator = functools.partial(collate_fn, tokenizer=tokenizer)
    )

    trainer.train()

    # save model
    save_model_id = 'multilabel_llama-3.1-8b'
    trainer.model.save_pretrained(save_model_id)
    tokenizer.save_pretrained(save_model_id)
