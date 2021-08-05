import sys
import os
import shutil
import time
import json
from typing import List, Union

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
import numpy as np
from torchmetrics import Accuracy, Precision, Recall
from Metrics import AreaUnderPrecisionCurve, TprFpr, Time1000Samples, ModifiedF1, SkAucRoc
from torchvision.transforms.transforms import  ToPILImage, ToTensor
from DataAugement import RandAugment

from DataLoaders import load_gen_data_dir
from LossFunctions import softmax_mse_loss, symmetric_mse_loss
from Models import cifar_shakeshake26, mt_shake_shake_params

import warnings
warnings.filterwarnings("ignore")

DATASETS_DIR = "Datasets"
OUT_DIR = "outputs"
NO_LABEL = -1


def eval_model(model, model_name, val_data_loaders, classes, fold_num, dataset_name, model_2=None, model2_name=None,
               out_dir=OUT_DIR):
    """
    Evaluates performance of up to 2 models
    :param model:
    :param model_name:
    :param val_data_loaders:
    :param classes:
    :param fold_num:
    :param dataset_name:
    :param model_2:
    :param model2_name:
    :param out_dir:
    :return: dictionary of evaluation restuls
    """
    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL)
    loss_model_1 = torch.Tensor([0])
    loss_model_2 = torch.Tensor([0])
    infer_time = (Time1000Samples(model), Time1000Samples(model_2))
    all_metrics = {"accuracy": (Accuracy(), Accuracy()),
                   "F1": (ModifiedF1(classes=classes), ModifiedF1(classes=classes)),
                   "stats": (TprFpr(classes), TprFpr(classes)),
                   "recall": (Recall(), Recall()),
                   "precision": (Precision(), Precision()),
                   "auroc": (SkAucRoc(classes=classes), SkAucRoc(classes=classes)),
                   "area_under_precision_recall": (AreaUnderPrecisionCurve(classes, out_dir, fold_num,
                                                                           model_name, dataset_name),
                                                   AreaUnderPrecisionCurve(classes, out_dir, fold_num,
                                                                           model_name, dataset_name)),
                   }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    if model_2 is not None:
        model_2.eval()
    for val_data_loader in val_data_loaders:
        for x, y_true in val_data_loader:
            x = torch.autograd.Variable(x).to(device)
            y_true = torch.autograd.Variable(y_true).to(device)
            preds = model(x)
            preds = preds[0] if type(preds) is tuple else preds
            batch_size = len(y_true)
            for metric in all_metrics.values():
                metric[0].update(preds=preds.detach().to('cpu'), target=y_true.detach().to('cpu'))
                loss_model_1 += (loss_func(preds.detach().to('cpu'), y_true.detach().to('cpu')) / batch_size)
                infer_time[0].update(x.detach().to('cpu'))
            if model_2 is not None:
                preds = model_2(x)
                preds = preds[0]
                for metric in all_metrics.values():
                    metric[1].update(preds=preds.detach().to('cpu'), target=y_true.detach().to('cpu'))
                    loss_model_2 += (loss_func(preds.detach().to('cpu'), y_true.detach().to('cpu')) / batch_size)
                    infer_time[1].update(x.detach().to('cpu'))

    ans = {
        f"loss_{model_name}": loss_model_1.numpy(),
        f"1000_samples_run_time_{model_name}": infer_time[0].compute()
    }
    if model_2 is not None:
        ans.update({
            f"loss_{model2_name}": loss_model_2.numpy(),
            f"1000_samples_run_time_{model2_name}": infer_time[0].compute()
        })

    for key, metric in all_metrics.items():
        try:
            ans[f"{key}_{model_name}"] = metric[0].compute().numpy()
        except AttributeError:
            ans[f"{key}_{model_name}"] = metric[0].compute()
        if model_2 is not None:
            try:
                ans[f"{key}_{model2_name}"] = metric[1].compute().numpy()
            except AttributeError:
                ans[f"{key}_{model_name}"] = metric[0].compute()

    return {fold_num: ans}


def dump_to_log(data, log=None):
    if log is None:
        print(data)
    else:
        try:
            if type(data) == dict:
                first_key = list(data.keys())[0]
                for key, value in data[first_key].items():
                    if type(data[first_key][key]) == np.ndarray:
                        data[first_key][key] = value.tolist()
        except (IndexError, AttributeError):
            pass

        with open(log, "a") as file:
            json.dump(data, file)
            file.write("\n")


# ------------ Ramp up function taken from: https://github.com/benathi/fastswa-semi-sup.

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return max(0., float(.5 * (np.cos(np.pi * current / rampdown_length) + 1)))


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

# -------------- End of Ramp up functions

def select_best_hyper_parameters(dataset_name, epochs):
    """
    Evaluate best hyper parameters for Fast-SWA and Mean-Teacher from orginal paper.
    Hyper parameters are taken from the list below using 3 fold cross validation
    :param dataset_name:
    :param epochs: number of epochs for pre-training the model
    :return: dictionary with best hyper parameters changes
    """
    print("------ Start hyper parameters search ----------")
    hyper_parameters = [
        {   # Original parameters
        },
        {
            "optimizer_args": {
                'lr': 0.1,
                "weight_decay": 2e-3,
            },
            'ema_decay': 0.93,
        },
        {
            "logit_distance_cost": 0.015,
            'ema_decay': 0.93,
            "consistency_rampup": 4,
        },
        {
            'ema_decay': 0.93,
            "consistency_rampup": 7,
            "consistency": 95.0,
            'fastswa_freq': '10',
        }
    ]
    accuracies = list()
    for idx, curr_test_params in enumerate(hyper_parameters):
        print(f"hyper parameters test - {idx}")
        curr_params = mt_shake_shake_params()
        curr_params.update(curr_test_params)
        accuracies.append(train_helper(dataset_name, 3, f"{OUT_DIR}/fast_swa_{dataset_name}_hyper_{idx}_log.log",
                                       curr_params, epochs))

    best_hp_idx = np.argmax(accuracies)
    dump_to_log(hyper_parameters[best_hp_idx], f"{OUT_DIR}/fast_swa_{dataset_name}_hyper_log.log")
    print(f"best hyper-parameters idx: {best_hp_idx},\nvalues: {hyper_parameters[best_hp_idx]}")
    print("------ END hyper parameters search ----------")
    return hyper_parameters[best_hp_idx]
    ######     Best hyperparameters index is 0 !!!!!!!!!


def main_original(idx_to_run):
    """
    Run hyper parameter search for Fast-SWA & Mean-Teacher from original paper on dataset shape.
    Then use best hyper parameters to run 10 fold cross validation on all data sets
    :return:
    """
    try:
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    except FileNotFoundError:
        pass
    os.makedirs(OUT_DIR, exist_ok=True)

    epochs = 60
    hp_dataset = "shapes"
    best_params = select_best_hyper_parameters(hp_dataset, epochs)
    params_to_use = mt_shake_shake_params()
    params_to_use.update(best_params)
    for dataset_name in os.listdir(DATASETS_DIR):
        print(f"Training on dataset - {dataset_name}")
        train_helper(dataset_name, 10, f"{OUT_DIR}/fast_swa_{dataset_name}_log.log", params_to_use, epochs)
    print("--------- END Train Fast-SWA -------------------")


def train_helper(dataset_name, k_fold, log_name, params, epochs) -> float:
    """
    Train original Fast-SWA
    :param dataset_name:
    :param k_fold:
    :param log_name: log file name
    :param params: hyper parameters to update
    :param epochs: pre-train epochs for base model
    :return:
    """
    ans = list()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unlabeled_gen, labeled_gen, classes = load_gen_data_dir(os.path.join(DATASETS_DIR, dataset_name), k_fold)
    for fold_num, ((unlabeled_train, unlabeled_val), (labeled_train, labeled_val)) in enumerate(zip(unlabeled_gen, labeled_gen)):
        base_model = cifar_shakeshake26(pretrained=False, num_classes=len(classes)).to(device)
        teacher_model = cifar_shakeshake26(pretrained=False, num_classes=len(classes)).to(device)
        swa_model = cifar_shakeshake26(pretrained=False, num_classes=len(classes)).to(device)
        mt = MeanTeacher(base_model, teacher_model, None, dataset_name, **params)
        fast_swa = FastSWA(mt, swa_model, log_name, dataset_name, **params)
        fast_swa.train_model_and_swa(unlabeled_train, labeled_train, epochs, fold_num)
        curr_ans = fast_swa.eval_swa([unlabeled_val, labeled_val], classes, fold_num)
        ans.append(curr_ans[fold_num]["accuracy_fast-swa"])

    return np.mean(ans)


def train_helper_augmented(dataset_name, k_fold, log_name, params, epochs) -> float:
    """
    Train the improved Fast-SWA model
    :param dataset_name:
    :param k_fold:
    :param log_name: log file name
    :param params: hyper parameters to update
    :param epochs: pre-train epochs for base model
    :return:
    """
    ans = list()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temp_loaders = [load_gen_data_dir(os.path.join(DATASETS_DIR, dataset_name), k_fold),
                    load_gen_data_dir(os.path.join(DATASETS_DIR, dataset_name), k_fold),
                    load_gen_data_dir(os.path.join(DATASETS_DIR, dataset_name), k_fold)]
    classes = temp_loaders[0][-1]
    unlabeled_loaders = [curr_temp[0] for curr_temp in temp_loaders]
    labeled_loaders = [curr_temp[1] for curr_temp in temp_loaders]

    for fold_num, curr_data_loaders in enumerate(zip(*unlabeled_loaders, *labeled_loaders)):
        curr_unlabeled_loaders = curr_data_loaders[:len(curr_data_loaders)//2]
        curr_labeled_loaders = curr_data_loaders[len(curr_data_loaders)//2:]
        base_models = [cifar_shakeshake26(pretrained=False, num_classes=len(classes)).to(device)
                       for _ in range(0, 3)]
        fake_teachers = [cifar_shakeshake26(pretrained=False, num_classes=len(classes)).to(device)
                         for _ in range(0, 3)]
        teacher_model = cifar_shakeshake26(pretrained=False, num_classes=len(classes)).to(device)
        swa_model = cifar_shakeshake26(pretrained=False, num_classes=len(classes)).to(device)
        students = [MeanTeacher(base, teacher, None, dataset_name, augment_values=(2, 7), **params)
                    for base, teacher in zip(base_models, fake_teachers)]
        ms = MultipleStudents(students, teacher_model, **params)
        fast_swa = FastSWA(ms, swa_model, log_name, dataset_name, **params)
        fast_swa.train_model_and_swa(curr_unlabeled_loaders, curr_labeled_loaders, epochs, fold_num, use_augment=True)
        curr_ans = fast_swa.eval_swa([curr_unlabeled_loaders[0][1], curr_labeled_loaders[0][1]], classes, fold_num,
                                     "outputs_aug")
        ans.append(curr_ans[fold_num]["accuracy_fast-swa"])

    return np.mean(ans)


def select_best_hyper_parameters_augmented(dataset_name, epochs):
    """
    Evaluate best hyper parameters for our improvement to fast-SWA
    Hyper parameters are taken from the list below using 3 fold cross validation
    :param dataset_name:
    :param epochs: number of epochs for pre-training the model
    :return: dictionary with best hyper parameters changes
    """
    print("------ Start hyper parameters search ----------")
    accuracies = list()

    hyper_parameters = [
        {   # Original parameters
        },
        {
            "optimizer_args": {
                'lr': 0.1,
                "weight_decay": 2e-3,
            },
            'ema_decay': 0.93,
        },
        {
            "logit_distance_cost": 0.015,
            'ema_decay': 0.93,
            "consistency_rampup": 4,
        },
        {
            'ema_decay': 0.93,
            "consistency_rampup": 7,
            "consistency": 95.0,
            'fastswa_freq': '10',
        }
    ]
    for idx, curr_hyper in enumerate(hyper_parameters):
        print(f"hyper parameters test augmented - {idx}")
        params = mt_shake_shake_params()
        params.update(curr_hyper)
        k_fold = 3
        log_name = f"{OUT_DIR}/fast_swa_{dataset_name}_hyper_{idx}_log.log"
        accuracies.append(train_helper_augmented(dataset_name, k_fold, log_name, params, epochs))

    best_hp_idx = np.argmax(accuracies)
    dump_to_log(hyper_parameters[best_hp_idx], f"{OUT_DIR}/fast_swa_{dataset_name}_hyper_log.log")
    print(f"Best hyper-parameters accuracy result: {accuracies[best_hp_idx]}"
          f"\nbest hyper-parameters idx: {best_hp_idx},\nvalues: {hyper_parameters[best_hp_idx]}"
          "------ END hyper parameters search ----------")
    return hyper_parameters[best_hp_idx]


def main_augmented(idx_to_run):
    """
    Run hyper parameter search for Fast-SWA & Mean-Teacher from original paper on dataset shape.
    Then use best hyper parameters to run 10 fold cross validation on all data sets
    :return:
    """
    print("Running Augmented")
    try:
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    except FileNotFoundError:
        pass
    os.makedirs(OUT_DIR, exist_ok=True)

    epochs = 60
    hp_dataset = "shapes"
    best_params = select_best_hyper_parameters_augmented(hp_dataset, epochs)
    params = mt_shake_shake_params()
    params.update(best_params)
    for dataset_name in os.listdir("./Datasets"):
        print(f"current dataset being used {dataset_name}")
        k_fold = 10
        log_name = f"{OUT_DIR}/fast_swa_{dataset_name}_log.log"
        train_helper_augmented(dataset_name, k_fold, log_name, params, epochs)
        print("--------- END Train Fast-SWA -------------------")


class MeanTeacher:
    """
    Mean-Teache class. For trainign and evaluationg the Mean-Teacher model
    """
    def __init__(self, base_model: nn.Module, teacher_model: nn.Module, log_file_path: str, dataset_name: str,
                 lr_rampdown_epochs, epoch_args, cycle_interval, logit_distance_cost, ema_decay,
                 num_cycles, optimizer_args, consistency_rampup, consistency, optimizer=SGD, augment_values=(4, 10),
                 **kwargs):

        # Make teacher untrainable from its own loss
        for param in teacher_model.parameters():
            param.detach_()

        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._base_model = base_model
        self._teacher_model = teacher_model
        self.log_file = log_file_path
        self._optimizer = optimizer(self.base_model.parameters(), **optimizer_args)
        self.epoch = 0
        self.global_step = 0
        self.base_lr = optimizer_args['lr']
        self.lr_rampdown_epochs = lr_rampdown_epochs
        self.epoch_args = epoch_args
        self.cycle_interval = cycle_interval
        self.logit_distance_cost = logit_distance_cost
        self.ema_decay = ema_decay
        self.num_cycles = num_cycles
        self.consistency_rampup = consistency_rampup
        self.consistency = consistency
        self.img_augment = RandAugment(*augment_values)

    def train(self, data_loader, epochs, val_data_loader, classes, fold_num):
        """
        Train the Mean-Teacher for the amount of epochs and evaluate at the end of each epoch
        :param data_loader:
        :param epochs:
        :param val_data_loader:
        :param classes:
        :param fold_num:
        :return:
        """
        total_epochs = epochs + self.cycle_interval*self.num_cycles
        print(f"will run Mean Teacher for {total_epochs} epochs")
        self.dump_to_log(f"will run Mean Teacher for {total_epochs} epochs\nRunning On Dataset: {self.dataset_name}")
        start_time = time.time()
        for _ in range(0, total_epochs):
            loss, lr = self.train_single_epoch(data_loader)
            eval_ans = self.eval_mt_model(val_data_loader, classes=classes, fold_num=fold_num,
                                          dataset_name=self.dataset_name)
            eval_ans.update({"loss": loss, "lr": lr})
            self.dump_to_log({self.epoch: eval_ans})
            self.epoch += 1
        end_time = time.time()
        self.dump_to_log(f"Train time for: {self.dataset_name} is: {end_time - start_time}\n"
                         f"----------------------------------\n----------------------------------")

    def _single_train_helper(self, inp, labels, use_augment=False):
        """
        Actual training of the student model for a single batch
        :param inp:
        :param labels:
        :return: loss on batch
        """
        batch_size = len(inp)
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).to(self.device)
        consistency_criterion = softmax_mse_loss
        residual_logit_criterion = symmetric_mse_loss

        if not use_augment:
            input_var_st = torch.autograd.Variable(inp).to(self.device)
            input_var_teacher = torch.autograd.Variable(inp).to(self.device)
        else:
            # Dataloaders must return a tensor, therfore we need to convert back to PIL Image here to allow the
            # RandAugment transformations
            imgs = [ToPILImage()(curr) for curr in inp]
            aug_1 = self.img_augment.transform_list(imgs)
            aug_2 = self.img_augment.transform_list(imgs)
            inp_1 = torch.cat([torch.unsqueeze(ToTensor()(curr), dim=0) for curr in aug_1])
            inp_2 = torch.cat([torch.unsqueeze(ToTensor()(curr), dim=0) for curr in aug_2])
            input_var_st = torch.autograd.Variable(inp_1).to(self.device)
            input_var_teacher = torch.autograd.Variable(inp_2).to(self.device)

        if labels is not None:
            labels = torch.autograd.Variable(labels).to(self.device)

        model_out_1, model_out_2 = self._base_model(input_var_st)
        teacher_out, _ = self._teacher_model(input_var_teacher)
        teacher_out = Variable(teacher_out.detach().data, requires_grad=False)

        class_loss = class_criterion(model_out_1, labels) / batch_size if labels is not None else 0
        res_loss = self.logit_distance_cost * residual_logit_criterion(model_out_1, model_out_2) / batch_size
        consistency_weight = self.get_current_consistency_weight()
        consistency_loss = consistency_weight * consistency_criterion(model_out_2, teacher_out) / batch_size
        loss = class_loss + consistency_loss + res_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss

    def train_single_epoch(self, unlabeled_data_loader, labeled_data_loader, use_augment=False):
        """
        Train the studen model for a single epoch and update the teacher at the end of the epoch
        :param use_augment:
        :param unlabeled_data_loader:
        :param labeled_data_loader:
        :return:
        """
        full_loss = 0
        self._base_model.train()
        self._teacher_model.train()

        steps_per_epoch = len(unlabeled_data_loader) + len(labeled_data_loader)
        for curr_step_in_unlabeled, (inp, _) in enumerate(unlabeled_data_loader):
            self.adjust_learning_rate(curr_step_in_unlabeled, steps_per_epoch)
            full_loss += self._single_train_helper(inp, None, use_augment)

        for curr_step_in_labeled, (inp, label) in enumerate(labeled_data_loader):
            self.adjust_learning_rate(curr_step_in_labeled + len(unlabeled_data_loader), steps_per_epoch)
            full_loss += self._single_train_helper(inp, label, use_augment)

        self.global_step += 1
        self.update_teacher_variables()
        print(f"loss {full_loss}")
        return full_loss

    def update_teacher_variables(self):
        """
        Updat the mean teacher - based on code from: https://github.com/benathi/fastswa-semi-sup.
        :return:
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_decay)
        for ema_param, param in zip(self._teacher_model.parameters(), self._base_model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def eval_mt_model(self, val_data_loader, classes, fold_num, dataset_name):
        return eval_model(self._base_model, "base_model", val_data_loader, classes=classes,
                          fold_num=fold_num, dataset_name=dataset_name,
                          model_2=self._teacher_model, model2_name="teacher")

    def get_current_consistency_weight(self):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * sigmoid_rampup(self.epoch, self.consistency_rampup)

    def dump_to_log(self, data):
        dump_to_log(data, self.log_file)

    def adjust_learning_rate(self, curr_step, steps_per_epoch):
        """
        Calculate learning rate at each step based on code from: https://github.com/benathi/fastswa-semi-sup.
        :param curr_step:
        :param steps_per_epoch:
        :return:
        """
        lr = self.base_lr
        part_epoch = self.epoch + curr_step/steps_per_epoch
        if self.lr_rampdown_epochs:
            if part_epoch < self.epoch_args:
                lr *= cosine_rampdown(part_epoch, self.lr_rampdown_epochs)
            else:
                lr_rampdown_epochs = self.lr_rampdown_epochs
                lr *= cosine_rampdown(
                    (lr_rampdown_epochs - (self.lr_rampdown_epochs - self.epoch_args) - self.cycle_interval) + (
                                (part_epoch - self.epoch_args) % self.cycle_interval),
                    lr_rampdown_epochs)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    @property
    def base_model(self):
        return self._base_model

    @property
    def teacher_model(self):
        return self._teacher_model

    @property
    def optimizer(self):
        return self._optimizer

    def get_learn_model(self):
        return self._base_model


class MultipleStudents:
    """
    Class for Mean-Teacher upgarde with multiple students
    """
    def __init__(self, students: List[MeanTeacher], teacher, ema_decay, **kwargs):
        self.students = students
        self.teacher = teacher
        self.global_step = 0
        self.ema_decay = ema_decay

    def train_single_epoch(self, unlabeled_data_loaders: List, labeled_data_loaders: List, use_augment: bool):
        """
        Train each student for a single epoch and finally update the mean teacher
        :param unlabeled_data_loaders:
        :param labeled_data_loaders:
        :return:
        """
        for student, unlabeled_data_loader, labeled_data_loader in zip(self.students, unlabeled_data_loaders,
                                                                       labeled_data_loaders):
            student.train_single_epoch(unlabeled_data_loader[0], labeled_data_loader[0], use_augment=use_augment)
        self.global_step += 1
        self.update_teacher()

    def update_teacher(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_decay)
        for student in self.students:
            for ema_param, param in zip(self.teacher.parameters(), student.base_model.parameters()):
                ema_param.data.mul_(alpha).add_(1 - alpha, param.data/len(self.students))

    @property
    def teacher_model(self):
        return self.teacher

    def get_learn_model(self):
        return self.teacher


class FastSWA:
    """
    Fast-SWA model as described in original papaer
    """
    def __init__(self, mt: Union[MeanTeacher, MultipleStudents], swa_model: nn.Module, log_path,  dataset_name: str,
                 num_cycles, cycle_interval, fastswa_freq, epoch_args, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for param in swa_model.parameters():
            param.detach_()

        self.dataset_name = dataset_name
        self.mt = mt
        self.teacher_model = mt.teacher_model
        self.swa_model = swa_model
        self.num_params = 0
        self.cycle_interval = cycle_interval
        self.num_cycles = num_cycles
        self.log_file = log_path
        self.fastswa_freq = int(fastswa_freq)
        self.epoch = 0
        self.epoch_args = epoch_args
        self.updated_swa = 0

    def update_fast_swa(self):
        """
        Update the Fast-SWA model weights
        :return:
        """
        self.num_params += 1
        if self.num_params == 1:
            self.swa_model.load_state_dict(self.mt.get_learn_model().state_dict())
        else:
            inv = 1./float(self.num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), self.teacher_model.parameters()):
                swa_p.data.add_(-inv*swa_p.data)
                swa_p.data.add_(inv*src_p.data)

    def reset(self):
        self.num_params = 0

    def train_model_and_swa(self, unlabeled_data_loader, labeled_data_loader, epochs, fold_num,
                            use_augment=False):
        """
        Pre-traind the student model of the Mean-Teacher, then continues training and updates the
        Fast-SWA model when needed
        :param unlabeled_data_loader:
        :param labeled_data_loader:
        :param epochs:
        :param fold_num:
        :return:
        """
        total_epochs = epochs + self.cycle_interval*self.num_cycles
        print(f"will run Fast SWA for {total_epochs} epochs")
        self.dump_to_log(f"will run Fast SWA for {total_epochs} epochs\n"
                         f"Running on Dataset: {self.dataset_name}\nRunning on fold: {fold_num}")
        start_time = time.time()
        for epoch in range(0, total_epochs):
            self.mt.train_single_epoch(unlabeled_data_loader, labeled_data_loader, use_augment=use_augment)
            if epoch >= epochs - self.cycle_interval and\
                    (epoch - self.epoch_args + self.cycle_interval) % self.fastswa_freq == 0:
                print("update swa")
                self.updated_swa += 1
                self.update_fast_swa()
                if type(unlabeled_data_loader) is list or type(unlabeled_data_loader) is tuple:
                    self.update_batchnorm(unlabeled_data_loader[0][0])
                else:
                    self.update_batchnorm(unlabeled_data_loader)
            self.epoch += 1
            print(f"Finished Epoch {self.epoch}")
        end_time = time.time()
        self.dump_to_log({fold_num: {
            "updated_swa": self.updated_swa,
            "train_time": end_time - start_time}})

    def eval_swa(self, val_data_loaders, classes, fold_num, out_dir = OUT_DIR):
        ans = eval_model(self.swa_model, "fast-swa", val_data_loaders, classes=classes,
                         fold_num=fold_num, dataset_name=self.dataset_name, out_dir=out_dir)
        self.dump_to_log(ans)
        return ans

    def dump_to_log(self, data):
        dump_to_log(data, self.log_file)

    def update_batchnorm(self, data_loader, steps_to_run=100):
        """
        Updates the fast-SWA bathcnorm parameters
        :param data_loader:
        :param steps_to_run:
        :return:
        """
        self.swa_model.train()
        for idx, (x, y) in enumerate(data_loader):
            if idx > steps_to_run:
                return
            input_var = torch.autograd.Variable(x, volatile=True).to(self.device)
            target_var = torch.autograd.Variable(y, volatile=True).to(self.device)
            model_out = self.swa_model(input_var)


if __name__ == '__main__':
    augment = 1 if sys.argv[1] != "0" else 0
    if augment == 0:
        main_original()
    else:
        OUT_DIR = "outputs_aug"
        main_augmented(int(sys.argv[1]))
