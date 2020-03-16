import os
import random
import datetime

import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch
from torch import optim
from torch.utils import data
from pytorch_pretrained_bert import optimization
from pytorch_pretrained_bert import modeling
from pytorch_pretrained_bert import tokenization

import feature_processors
import metrics
from data_utils import data_processors
import bert
import random
import models
from models import utils


class BertSeq2seqModel:
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
            params["bert_model"],
            do_lower_case=params["lower_case"],
        )
        self.model = bert.BertForSeq2Seq.from_pretrained(
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
            X_train,
            y_train,
            split_name="train",
        )
        if validation_data is not None:
            X_valid, y_valid = validation_data
            dev_examples = data_processors.create_examples(
                X_valid,
                y_valid,
                split_name="dev",
            )
        else:
            dev_examples = None

        if best_model_output is not None:
            best_model_output = os.path.join(
                self.params["output_dir"],
                best_model_output,
            )

        self.params["num_train_epochs"] = n_epochs
        if batch_size is not None:
            self.params["train_batch_size"] = batch_size
        train_steps_per_epoch = int(
            len(train_examples) / self.params["train_batch_size"],
        )

        best_epoch_result = None
        for epoch_num in range(int(self.params["num_train_epochs"])):
            print("\nEpoch: {}".format(epoch_num + 1))
            self.model, result = train_one_seq2seq_epoch(
                self.model,
                self.tokenizer,
                self.params,
                train_examples,
                dev_examples,
                epoch_num + 1,
            )
            print(result)
            if validation_data is not None:
                if (
                    best_epoch_result is None
                    or result["accuracy_strict"] > best_epoch_result["accuracy_strict"]
                ):
                    best_epoch_result = result
                    best_epoch_result["best_epoch"] = epoch_num + 1
                    if best_model_output is not None:
                        torch.save(self.model.state_dict(), best_model_output)
                        best_epoch_result["model_filepath"] = best_model_output
        if best_epoch_result is None:
            return result
        return best_epoch_result


def evaluate_seq2seq_model(
    params,
    model,
    tokenizer,
    X_eval,
    y_eval,
    batch_size=None,
    verbose=True,
):
    eval_examples = data_processors.create_examples(
        X_eval,
        y_eval,
        split_name="test",
    )
    return evaluate_seq2seq(model, tokenizer, params, eval_examples, verbose)


def convert_labels_to_tensor(labels):
    max_num_labels = max([len(label_list) for label_list in labels])
    object_labels = np.zeros((len(labels), max_num_labels + 2))
    for i, label_list in enumerate(labels):
        object_labels[i][0] = models.dict.BOS
        for j, label in enumerate(label_list):
            object_labels[i][j + 1] = label + 4
        object_labels[i][len(label_list) + 1] = models.dict.EOS
    return torch.tensor(object_labels, dtype=torch.long)


def train_one_seq2seq_epoch(
    model,
    tokenizer,
    params,
    train_examples,
    valid_examples=None,
    epoch_num=-1,
):
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    train_features = feature_processors.convert_examples_to_features(
        train_examples,
        params["max_seq_length"],
        tokenizer,
    )
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features],
        dtype=torch.long,
    )
    all_label_ids = convert_labels_to_tensor(
        [f.label_id for f in train_features],
    )
    train_data = data.TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    train_sampler = data.RandomSampler(train_data)
    train_dataloader = data.DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=params["train_batch_size"],
    )

    model.train()
    tr_loss, nb_tr_steps = 0, 0
    report_total = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(params["device"]) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        outputs, targets = model(
            input_ids,
            label_ids.transpose(1, 0),
            segment_ids,
            input_mask,
        )
        loss, num_total = model.compute_loss(outputs, targets)
        report_total += num_total.item()

        loss.backward()
        tr_loss += loss.item()
        nb_tr_steps += 1
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % 300 == 0:
            train_result = {
                "train_log_loss": tr_loss / nb_tr_steps,
                "train_log_loss_cmp": tr_loss / report_total,
            }
            print(train_result)
            torch.save(
                model.state_dict(),
                os.path.join(params["output_dir"], "model_checkpoint.pth"),
            )
            if valid_examples is not None:
                valid_result, _ = evaluate_seq2seq(
                    model,
                    tokenizer,
                    params,
                    valid_examples,
                )
                print("==Validation==", step)
                print(valid_result)
                total_result = {**train_result, **valid_result}
                total_result_csv = ",".join(
                    np.array(
                        [epoch_num, step + 1] + list(total_result.values()),
                        dtype=str,
                    ),
                )
                with open(params["logfile"], "a") as f:
                    f.write(total_result_csv + "\n")

                model.train()

    train_result = {
        "train_log_loss": tr_loss / nb_tr_steps,
        "train_log_loss_cmp": tr_loss / report_total,
    }
    if valid_examples is not None:
        valid_result, _ = evaluate_seq2seq(
            model,
            tokenizer,
            params,
            valid_examples,
        )
        model.train()

    total_result = {**train_result, **valid_result}
    total_result_csv = ",".join(
        np.array(
            [epoch_num, nb_tr_steps] + list(total_result.values()),
            dtype=str,
        ),
    )
    with open(params["logfile"], "a") as f:
        f.write(total_result_csv + "\n")
    return model, total_result


def convertToLabels(s, stop):
    labels = []
    for label in s:
        if label.item() == stop:
            break
        labels.append(label.item() - 4)
    return sorted(labels)


def evaluate_seq2seq(model, tokenizer, params, valid_examples, verbose=True):
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    num_train_optimization_steps = int(
        len(valid_examples) / params["train_batch_size"],
    )

    train_features = feature_processors.convert_examples_to_features(
        valid_examples,
        params["max_seq_length"],
        tokenizer,
    )

    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features],
        dtype=torch.long,
    )
    all_label_ids = [f.label_id for f in train_features]

    train_data = data.TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        convert_labels_to_tensor(all_label_ids),
    )
    train_dataloader = data.DataLoader(
        train_data,
        batch_size=params["eval_batch_size"],
    )

    model.eval()
    tr_loss, nb_tr_steps = 0, 0
    report_total = 0
    reference, candidate, alignments = all_label_ids, [], []
    iter_data = enumerate(train_dataloader)
    if verbose:
        iter_data = enumerate(tqdm(train_dataloader, desc="Iteration"))

    preds = []
    for step, batch in iter_data:
        batch = tuple(t.to(params["device"]) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        outputs, targets = model(
            input_ids,
            label_ids.transpose(1, 0),
            segment_ids,
            input_mask,
        )

        loss, num_total = model.compute_loss(outputs, targets)

        report_total += num_total.item()
        tr_loss += loss.item()
        nb_tr_steps += 1
        if (step + 1) % 200 == 0:
            train_result = {
                "val_log_loss": tr_loss / nb_tr_steps,
                "val_log_loss_cmp": tr_loss / report_total,
            }
            print(train_result)

        samples, alignment, batch_probs = model.beam_sample(
            input_ids,
            input_mask,
            segment_ids,
            beam_size=5,
        )
        preds += batch_probs
        candidate += [convertToLabels(s, models.dict.EOS) for s in samples]

        alignments += [align for align in alignment]

    train_result = {
        "val_log_loss": tr_loss / nb_tr_steps,
        "val_log_loss_cmp": tr_loss / report_total,
    }
    result = utils.eval_metrics(reference, candidate, params)
    return {**train_result, **result}, candidate, np.array(preds)


def train_one_epoch(
    model,
    tokenizer,
    params,
    train_examples,
    valid_examples=None,
):
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    train_features = feature_processors.convert_examples_to_features(
        train_examples,
        params["max_seq_length"],
        tokenizer,
    )

    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features],
        dtype=torch.long,
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in train_features],
        dtype=torch.long,
    )
    train_data = data.TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    train_sampler = data.RandomSampler(train_data)
    train_dataloader = data.DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=params["train_batch_size"],
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

    train_result = {"train_log_loss": tr_loss / nb_tr_steps}
    if valid_examples is not None:
        valid_result, valid_prob_preds = evaluate(
            model,
            tokenizer,
            params,
            valid_examples,
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
        valid_examples,
        params["max_seq_length"],
        tokenizer,
    )
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features],
        dtype=torch.long,
    )
    eval_data = data.TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
    )

    eval_sampler = data.SequentialSampler(eval_data)
    eval_dataloader = data.DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=params["eval_batch_size"],
    )

    model.eval()
    softmax = torch.nn.Softmax(dim=-1)

    if multitask:
        test_preds = [[] for _ in range(len(params["num_labels"]))]
    else:
        test_preds = []
    for input_ids, input_mask, segment_ids in tqdm(
        eval_dataloader,
        desc="Predicting",
    ):
        logits = model(
            input_ids.to(params["device"]),
            segment_ids.to(params["device"]),
            input_mask.to(params["device"]),
        )
        if multitask:
            logits = [task_logits.detach().cpu() for task_logits in logits]
            for task_num, task_logits in enumerate(logits):
                test_preds[task_num] += list(softmax(task_logits).numpy())
        else:
            logits = logits.detach().cpu()
            test_preds += list(softmax(logits).numpy())
    if multitask:
        test_preds = [np.array(preds) for preds in test_preds]
    return test_preds


def get_representations(model, tokenizer, params, valid_examples):
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    eval_features = feature_processors.convert_examples_to_features(
        valid_examples,
        params["max_seq_length"],
        tokenizer,
    )
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features],
        dtype=torch.long,
    )
    eval_data = data.TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
    )

    eval_sampler = data.SequentialSampler(eval_data)
    eval_dataloader = data.DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=params["eval_batch_size"],
    )

    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    representations = []
    encoded_layers_list = []
    for input_ids, input_mask, segment_ids in tqdm(
        eval_dataloader,
        desc="Predicting",
    ):
        encoded_layers, encoder_outputs = model.get_representations(
            input_ids.to(params["device"]),
            segment_ids.to(params["device"]),
            input_mask.to(params["device"]),
        )
        encoder_outputs = encoder_outputs.detach().cpu()
        encoded_layers = np.array(
            [encoded_layer.detach().cpu().numpy() for encoded_layer in encoded_layers],
        )
        representations += list(encoder_outputs.numpy())
        encoded_layers_list.append(encoded_layers)
    return np.array(representations), encoded_layers_list


def evaluate(model, tokenizer, params, valid_examples):
    print("***** Running evaluation *****")

    prob_preds = predict(model, tokenizer, params, valid_examples)
    true_labels = np.array(
        [int(example.label) for i, example in enumerate(valid_examples)],
    )
    result = {
        "eval_log_loss": metrics.log_loss(true_labels, prob_preds),
        "eval_accuracy": metrics.accuracy(true_labels, prob_preds),
    }
    return result, prob_preds
