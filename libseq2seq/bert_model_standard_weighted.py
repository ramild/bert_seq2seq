import os
import random
import datetime

import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
from pytorch_pretrained_bert import optimization
from pytorch_pretrained_bert import modeling
from pytorch_pretrained_bert import tokenization

import feature_processors
import metrics
from data_utils import data_processors
import bert
from random import shuffle


class BertTextClassificationModel:
    def __init__(self, params):
        self.params = params
        if not os.path.exists(self.params["output_dir"]):
            os.makedirs(self.params["output_dir"])
        self.params["logfile"] = os.path.join(
            self.params["output_dir"],
            datetime.datetime.now().strftime("log_%d_%B_%Y_%I:%M%p.txt"),
        )
        print("Downloading BERT...")
        self.tokenizer = tokenization.BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lower_case"]
        )
        self.model = bert.BertForSequenceClassification.from_pretrained(
            params["bert_model"],
            cache_dir=params["cache_dir"],
            num_labels=params["num_labels"],
        ).to(params["device"])
        print("Completed!")

    def fit(
        self,
        X_train,
        y_train,
        batch_size=None,
        n_epochs=1,
        validation_data=None,
        best_model_output=None,
    ):
        train_examples = data_processors.create_examples(
            X_train, y_train, split_name="train"
        )
        shuffle(train_examples)
        if validation_data is not None:
            X_valid, y_valid = validation_data
            dev_examples = data_processors.create_examples(
                X_valid, y_valid, split_name="dev"
            )
            shuffle(dev_examples)
        else:
            dev_examples = None

        if best_model_output is not None:
            best_model_output = os.path.join(
                self.params["output_dir"], best_model_output
            )

        self.params["num_train_epochs"] = n_epochs
        if batch_size is not None:
            self.params["train_batch_size"] = batch_size
        train_steps_per_epoch = int(
            len(train_examples) / self.params["train_batch_size"]
        )

        best_epoch_result = None
        for epoch_num in range(int(self.params["num_train_epochs"])):
            print("\nEpoch: {}".format(epoch_num + 1))
            self.model, result = train_one_epoch(
                self.model, self.tokenizer, self.params, train_examples, dev_examples
            )
            print(result)
            if validation_data is not None:
                if (
                    best_epoch_result is None
                    or result["eval_log_loss"] < best_epoch_result["eval_log_loss"]
                ):
                    best_epoch_result = result
                    best_epoch_result["best_epoch"] = epoch_num + 1
                    if best_model_output is not None:
                        torch.save(self.model.state_dict(), best_model_output)
                        best_epoch_result["model_filepath"] = best_model_output
        if best_epoch_result is None:
            return result
        return best_epoch_result

    def predict(self, X_eval, y_eval=None, batch_size=None):
        if y_eval is None:
            y_eval = np.zeros(len(X_eval), dtype=int)
        eval_examples = data_processors.create_examples(
            X_eval, y_eval, split_name="eval"
        )
        if batch_size is not None:
            self.params["eval_batch_size"] = batch_size
        return predict(self.model, self.tokenizer, self.params, eval_examples)

    def evaluate(self, X_eval, y_eval, batch_size=None):
        eval_examples = data_processors.create_examples(
            X_eval, y_eval, split_name="test"
        )
        if batch_size is not None:
            self.params["eval_batch_size"] = batch_size
        return evaluate(self.model, self.tokenizer, self.params, eval_examples)

    def load(self, model_filepath):
        self.model.load_state_dict(torch.load(model_filepath))

    def state_dict(self):
        return self.model.state_dict()


class BertMultiTaskTextClassificationModel:
    def __init__(self, params):
        self.params = params
        if not os.path.exists(self.params["output_dir"]):
            os.makedirs(self.params["output_dir"])
        self.params["logfile"] = os.path.join(
            self.params["output_dir"],
            datetime.datetime.now().strftime("log_%d_%B_%Y_%I:%M%p.txt"),
        )
        print("Downloading BERT...")
        self.tokenizer = tokenization.BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lower_case"]
        )
        self.model = bert.BertForMultiTaskSequenceClassification.from_pretrained(
            params["bert_model"],
            cache_dir=params["cache_dir"],
            tasks_num_labels=params["num_labels"],
            device=params["device"],
        ).to(params["device"])
        print("Completed!")

    def fit(
        self,
        X_train,
        y_train,
        batch_size=None,
        n_epochs=1,
        validation_data=None,
        best_model_output=None,
    ):
        train_examples = data_processors.create_multitask_examples(
            X_train, y_train, split_name="train"
        )
        shuffle(train_examples)
        if validation_data is not None:
            X_valid, y_valid = validation_data
            dev_examples = data_processors.create_multitask_examples(
                X_valid, y_valid, split_name="dev"
            )
            shuffle(dev_examples)
        else:
            dev_examples = None

        if best_model_output is not None:
            best_model_output = os.path.join(
                self.params["output_dir"], best_model_output
            )

        self.params["num_train_epochs"] = n_epochs
        if batch_size is not None:
            self.params["train_batch_size"] = batch_size
        train_steps_per_epoch = int(
            len(train_examples) / self.params["train_batch_size"]
        )
        num_train_optimization_steps = (
            train_steps_per_epoch * self.params["num_train_epochs"]
        )

        best_epoch_result = None
        for epoch_num in range(int(self.params["num_train_epochs"])):
            print("\nEpoch: {}".format(epoch_num + 1))
            self.model, result = train_one_epoch_multitask(
                self.model, self.tokenizer, self.params, train_examples, dev_examples
            )
            print(result)
            if validation_data is not None:
                if (
                    best_epoch_result is None
                    or result["eval_log_loss"] < best_epoch_result["eval_log_loss"]
                ):
                    best_epoch_result = result
                    best_epoch_result["best_epoch"] = epoch_num + 1
                    if best_model_output is not None:
                        torch.save(self.model.state_dict(), best_model_output)
                        best_epoch_result["model_filepath"] = best_model_output
        if best_epoch_result is None:
            return result
        return best_epoch_result

    def predict(self, X_eval, y_eval=None, batch_size=None):
        if y_eval is None:
            y_eval = np.zeros(len(X_eval), dtype=int)
        eval_examples = data_processors.create_multitask_examples(
            X_eval, y_eval, split_name="eval"
        )
        if batch_size is not None:
            self.params["eval_batch_size"] = batch_size
        return predict(self.model, self.tokenizer, self.params, eval_examples)

    def evaluate(self, X_eval, y_eval, batch_size=None):
        eval_examples = data_processors.create_multitask_examples(
            X_eval, y_eval, split_name="test"
        )
        if batch_size is not None:
            self.params["eval_batch_size"] = batch_size
        return evaluate_multitask(
            self.model, self.tokenizer, self.params, eval_examples
        )

    def load(self, model_filepath):
        self.model.load_state_dict(torch.load(model_filepath))

    def state_dict(self):
        return self.model.state_dict()

    def get_linear_weights(self):
        self.model.get_linear_weights()


def train_one_epoch_multitask(
    model, tokenizer, params, train_examples, valid_examples=None
):
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    num_train_optimization_steps = int(len(train_examples) / params["train_batch_size"])

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optimization.BertAdam(
        optimizer_grouped_parameters,
        lr=params["learning_rate"],
        warmup=params["warmup_proportion"],
        t_total=num_train_optimization_steps,
    )

    train_features = feature_processors.convert_examples_to_features(
        train_examples, params["max_seq_length"], tokenizer
    )

    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_task_ids = torch.tensor([f.task for f in train_features], dtype=torch.long)
    train_data = data.TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_task_ids
    )
    train_sampler = data.RandomSampler(train_data)
    train_dataloader = data.DataLoader(
        train_data, sampler=train_sampler, batch_size=params["train_batch_size"]
    )

    model.train()
    tr_loss, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(params["device"]) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, task_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids, task_ids)
        loss.backward()
        tr_loss += loss.item()
        nb_tr_steps += 1
        optimizer.step()
        optimizer.zero_grad()

    train_result = {
        "train_log_loss": tr_loss / nb_tr_steps,
    }
    if valid_examples is not None:
        valid_result, valid_prob_preds = evaluate_multitask(
            model, tokenizer, params, valid_examples
        )
        model.train()

    with open(params["logfile"], "a") as f:
        f.write(str({**train_result, **valid_result}) + "\n\n")

    return model, {**train_result, **valid_result}


def train_one_epoch(model, tokenizer, params, train_examples, valid_examples=None):
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    num_train_optimization_steps = int(len(train_examples) / params["train_batch_size"])

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optimization.BertAdam(
        optimizer_grouped_parameters,
        lr=params["learning_rate"],
        warmup=params["warmup_proportion"],
        t_total=num_train_optimization_steps,
    )

    train_features = feature_processors.convert_examples_to_features(
        train_examples, params["max_seq_length"], tokenizer
    )

    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = data.TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    train_sampler = data.RandomSampler(train_data)
    train_dataloader = data.DataLoader(
        train_data, sampler=train_sampler, batch_size=params["train_batch_size"]
    )

    model.train()
    tr_loss, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(params["device"]) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        loss.backward()
        tr_loss += loss.item()
        nb_tr_steps += 1
        optimizer.step()
        optimizer.zero_grad()

    train_result = {
        "train_log_loss": tr_loss / nb_tr_steps,
    }
    if valid_examples is not None:
        valid_result, valid_prob_preds = evaluate(
            model, tokenizer, params, valid_examples
        )
        model.train()

    with open(params["logfile"], "a") as f:
        f.write(str({**train_result, **valid_result}) + "\n\n")

    return model, {**train_result, **valid_result}


def predict(model, tokenizer, params, valid_examples, multitask=False):
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    eval_features = feature_processors.convert_examples_to_features(
        valid_examples, params["max_seq_length"], tokenizer
    )
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long
    )
    eval_data = data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    eval_sampler = data.SequentialSampler(eval_data)
    eval_dataloader = data.DataLoader(
        eval_data, sampler=eval_sampler, batch_size=params["eval_batch_size"]
    )

    model.eval()
    softmax = torch.nn.Softmax(dim=-1)

    if multitask:
        test_preds = [[] for _ in range(len(params["num_labels"]))]
    else:
        test_preds = []
    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Predicting"):
        logits = model(
            input_ids.to(params["device"]),
            segment_ids.to(params["device"]),
            input_mask.to(params["device"]),
        )
        logits = [task_logits.detach().cpu() for task_logits in logits]
        if multitask:
            for task_num, task_logits in enumerate(logits):
                test_preds[task_num] += list(softmax(task_logits).numpy())
        else:
            test_preds += list(softmax(logits).numpy())
    if multitask:
        test_preds = [np.array(preds) for preds in test_preds]
    return test_preds


def evaluate(model, tokenizer, params, valid_examples):
    print("***** Running evaluation *****")

    prob_preds = predict(model, tokenizer, params, valid_examples)
    true_labels = np.array(
        [int(example.label) for i, example in enumerate(valid_examples)]
    )
    result = {
        "eval_log_loss": metrics.log_loss(true_labels, prob_preds),
        "eval_accuracy": metrics.accuracy(true_labels, prob_preds),
    }
    return result, prob_preds


def evaluate_multitask(model, tokenizer, params, valid_examples):
    print("***** Running evaluation *****")

    prob_preds = predict(model, tokenizer, params, valid_examples, multitask=True)
    true_labels = np.array(
        [int(example.label) for i, example in enumerate(valid_examples)]
    )
    tasks = np.array([example.task for i, example in enumerate(valid_examples)])
    log_losses = []
    accuracies = []
    for task_num in range(len(params["num_labels"])):
        log_losses.append(
            metrics.log_loss(
                true_labels[tasks == task_num], prob_preds[task_num][tasks == task_num]
            )
        )
        accuracies.append(
            metrics.accuracy(
                true_labels[tasks == task_num], prob_preds[task_num][tasks == task_num]
            )
        )

    result = {"eval_log_loss": log_losses, "eval_accuracy": accuracies}
    return result, prob_preds
