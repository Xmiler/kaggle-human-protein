import matplotlib as plt
from functools import partial
import json

import numpy as np

import torch
from sklearn.metrics import f1_score
from livelossplot import PlotLosses


liveloss = PlotLosses()


def get_max_value(item):
    return np.array([item2[1] for item2 in item]).max()
def get_max_thr(item):
    ind = np.array([item2[1] for item2 in item]).argmax()
    return item[ind][0]

def calculate_f1s(gt, preds):
    return [('%.1f'%thr, f1_score(gt, preds > thr, average='macro')) for thr in np.arange(.1, 1.,.1)]
def calculate_accs(gt, preds):
    return [('%.1f'%thr, (gt == (preds > thr)).all(axis=1).sum()/gt.shape[0]) for thr in np.arange(.1, 1.,.1)]


class Trainer:
    def __init__(self, version, dataloader, model, optimizer, criterion):
        self._iter_num = -1

        self.version = version
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.batch_size = dataloader['train'].batch_size
        self.epoch_size = self.batch_size * len(dataloader['train'])

        self.statistics = {
            'batch': {'train': {'iter': [], 'lrs': [], 'loss': []}},
            'epoch': {'train': {'iter': [], 'loss': [], 'acc': [], 'f1_macro': []},
                      'val': {'iter': [], 'loss': [], 'acc': [], 'f1_macro': []}}
        }

    def get_iteration(self):
        return self._iter_num

    def plot_epoches(self):
        def get_max_values(item):
            return np.array([item2[1] for item2 in item]).max()

        def get_max_thr(item):
            ind = np.array([item2[1] for item2 in item]).argmax()
            return item[ind][0]

        # final
        print(' ========== VALIDATION SCORE ========== ')
        print('f1 macro : %.4f' % get_max_values(self.statistics['epoch']['val']['f1_macro'][-1]))
        print('threshold: %s' % get_max_thr(self.statistics['epoch']['val']['f1_macro'][-1]))
        print('epoch num: %d' % (self.statistics['epoch']['val']['iter'][-1] * self.batch_size / self.epoch_size))

        # f1_macro(max)
        plt.figure(figsize=(15, 8))
        for phase in ['train', 'val']:
            x_cnt = np.array(self.statistics['epoch'][phase]['iter'])
            x_cnt = x_cnt * self.batch_size / self.batch_size
            y_cnt = [get_max_values(item) for item in self.statistics['epoch'][phase]['f1_macro']]
            plt.plot(x_cnt, y_cnt, '.-', label=phase)
        plt.xlabel('epoch')
        plt.ylabel('f1_macro(max)')
        plt.legend()
        plt.grid()

        # f1_macro threshold
        plt.figure(figsize=(15, 8))
        for phase in ['train', 'val']:
            x_cnt = np.array(self.statistics['epoch'][phase]['iter'])
            x_cnt = x_cnt * self.batch_size / self.epoch_size
            y_cnt = [get_max_thr(item) for item in self.statistics['epoch'][phase]['f1_macro']]
            plt.plot(x_cnt, y_cnt, '.-', label=phase)
        plt.xlabel('epoch')
        plt.ylabel('f1_macro threshold')
        plt.legend()
        plt.grid()

        # loss
        plt.figure(figsize=(15, 8))
        for phase in ['train', 'val']:
            x_cnt = np.array(self.statistics['epoch'][phase]['iter'])
            x_cnt = x_cnt * self.batch_size / self.epoch_size
            y_cnt = self.statistics['epoch'][phase]['loss']
            plt.plot(x_cnt, y_cnt, '.-', label=phase)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()

        # acc
        plt.figure(figsize=(15, 8))
        for phase in ['train', 'val']:
            x_cnt = np.array(self.statistics['epoch'][phase]['iter'])
            x_cnt = x_cnt * self.batch_size / self.epoch_size
            y_cnt = [get_max_values(item) for item in self.statistics['epoch'][phase]['acc']]
            plt.plot(x_cnt, y_cnt, '.-', label=phase)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.grid()

        # acc threshold
        plt.figure(figsize=(15, 8))
        for phase in ['train', 'val']:
            x_cnt = np.array(self.statistics['epoch'][phase]['iter'])
            x_cnt = x_cnt * self.batch_size / self.epoch_size
            y_cnt = [get_max_thr(item) for item in self.statistics['epoch'][phase]['acc']]
            plt.plot(x_cnt, y_cnt, '.-', label=phase)
        plt.xlabel('epoch')
        plt.ylabel('acc threshold')
        plt.legend()
        plt.grid()

    def train_iteration(self, inputs, labels, lrs):
        self._iter_num += 1

        for i in [0, 1]:
            self.optimizer.param_groups[i]['lr'] = lrs[i]

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            if self._iter_num % int(self.epoch_size / self.batch_size / 10) == 0:
                self.statistics['batch']['train']['iter'].append(self.get_iteration())
                self.statistics['batch']['train']['lrs'].append(lrs)
                self.statistics['batch']['train']['loss'].append(loss.item())

        return outputs.sigmoid().cpu().data.numpy(), loss.item() * inputs.size(0)

    def _update_statistics_epoch(self, gts, preds, loss, phase):
        gts = np.concatenate(gts)
        preds = np.concatenate(preds)

        f1s = calculate_f1s(gts, preds)
        accs = calculate_accs(gts, preds)

        self.statistics['epoch'][phase]['iter'].append(self.get_iteration())
        self.statistics['epoch'][phase]['loss'].append(loss)
        self.statistics['epoch'][phase]['acc'].append(accs)
        self.statistics['epoch'][phase]['f1_macro'].append(f1s)

    def test_epoch(self, phase):
        loss_epoch = 0
        gt_epoch = []
        preds_epoch = []
        with torch.set_grad_enabled(False):
            for inputs, labels in self.dataloader[phase]:
                gt_epoch.append(labels.cpu().data.numpy())

                #                 inputs = inputs.to(device_id)
                #                 labels = labels.float().to(device_id)

                outputs = self.model(inputs)

                preds_epoch.append(outputs.sigmoid().cpu().data.numpy())

                loss = self.criterion(outputs, labels)

                loss_epoch += loss.item() * inputs.size(0)

        loss_epoch /= len(self.dataloader[phase].dataset)
        self._update_statistics_epoch(gt_epoch, preds_epoch, loss_epoch, phase)

    def train_epoch(self, lr_policy_fnc, lrs):
        lr_policy_fncs = [partial(lr_policy_fnc, lr_init=lr) for lr in lrs]
        loss_epoch = 0.
        gt_epoch = []
        preds_epoch = []
        with torch.set_grad_enabled(True):
            for iter_i, (inputs, labels) in enumerate(self.dataloader['train']):
                gt_epoch.append(labels.cpu().data.numpy())
                lrs = [fnc(iter_i) for fnc in lr_policy_fncs]
                preds, loss = self.train_iteration(inputs, labels, lrs)
                loss_epoch += loss
                preds_epoch.append(preds)

        loss_epoch /= len(self.dataloader['train'].dataset)
        self._update_statistics_epoch(gt_epoch, preds_epoch, loss_epoch, 'train')

        self.test_epoch('val')

        liveloss.update({
            'loss': self.statistics['epoch']['train']['loss'][-1],
            'f1_macro': get_max_value(self.statistics['epoch']['train']['f1_macro'][-1]),
            'val_loss': self.statistics['epoch']['val']['loss'][-1],
            'val_f1_macro': get_max_value(self.statistics['epoch']['val']['f1_macro'][-1])
        })
        liveloss.draw()

    def predict_test(self):
        preds = []
        with torch.set_grad_enabled(False):
            for inputs, _ in self.dataloader['test']:
                outputs = self.model(inputs)
                preds.append(outputs.sigmoid().cpu().data.numpy())
        preds = np.concatenate(preds)
        return preds

    def restore(self):
        self.model.load_state_dict(torch.load('%s.pth' % self.version))
        self.model.eval()
        self.statistics = json.load(Path('%s.stat.json' % self.version).open('r'))

    def store(self):
        torch.save(self.model.state_dict(), '%s.pth' % self.version)
        json.dump(trainer.statistics, Path('%s.stat.json' % self.version).open('w'))
