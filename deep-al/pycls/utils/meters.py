#This file is modified from official pycls repository to adapt in AL settings.
"""Meters."""

from collections import deque

import datetime
import numpy as np

from pycls.core.config import cfg
from pycls.utils.timer import Timer

import pycls.utils.logging as lu
import pycls.utils.metrics as metrics

import csv
from pathlib import Path


class BufferedFileLogger:
    def __init__(
            self,
            file_name,
            file_path='.',
            buffer_size=1000,
            header=("metric", "value", "global_step"),
        mode='a'
    ):
        self.file_path = Path(file_path)
        self.file_path.mkdir(parents=True, exist_ok=True)
        self.file_name = file_name
        self.buffer_size = buffer_size
        self.buffer = []

        # check if file exists
        if not (self.file_path / self.file_name).exists():
            mode = 'w'
            exists = False
        else:
            exists = True

        self.file = open(
            self.file_path / self.file_name,
            mode=mode,
            newline='',
            buffering=1  # Line buffering
        )

        self.writer = csv.writer(self.file)
        if not exists:
            # Write the header of the CSV file
            self.writer.writerow(header)

    def add_scalar(self, *args):
        self.buffer.append(args)
        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def _flush(self):
        if self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer = []

    def close(self):
        self._flush()
        self.file.close()



def eta_str(eta_td):
    """Converts an eta timedelta to a fixed-width string format."""
    days = eta_td.days
    hrs, rem = divmod(eta_td.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{0:02},{1:02}:{2:02}:{3:02}'.format(days, hrs, mins, secs)


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters, cur_iter, seed):
        self.epoch_iters = epoch_iters
        self.max_iter = cfg.OPTIM.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        self.cur_iter = cur_iter
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_samples = 0
        self.filelogger = BufferedFileLogger(
            file_name=f'training_metrics.csv',
            buffer_size=200,
            file_path=cfg.EXP_DIR,

            header=("al_budget", "epoch", "top1_err", "loss", 'lr')
        )

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.num_top1_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, loss, lr, mb_size):
        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_top1_mis += top1_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size


    def get_iter_stats(self, cur_epoch, cur_iter):
        eta_sec = self.iter_timer.average_time * (
            self.max_iter - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta_td = datetime.timedelta(seconds=int(eta_sec))
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            '_type': 'train_iter',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'iter': '{}/{}'.format(cur_iter + 1, self.epoch_iters),
            'top1_err': self.mb_top1_err.get_win_median(),
            'loss': self.loss.get_win_median(),
            'lr': self.lr,
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        lu.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        eta_sec = self.iter_timer.average_time * (
            self.max_iter - (cur_epoch + 1) * self.epoch_iters
        )
        eta_td = datetime.timedelta(seconds=int(eta_sec))
        mem_usage = metrics.gpu_mem_usage()
        top1_err = self.num_top1_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            '_type': 'train_epoch',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'top1_err': top1_err,
            'loss': avg_loss,
            'lr': self.lr,
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        lu.log_json_stats(stats)
        self.filelogger.add_scalar(self.cur_iter, int(stats["epoch"].split('/')[0]), stats['top1_err'], stats['loss'], stats['lr'])

    def close(self):
        self.filelogger.close()


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, max_iter, cur_iter, seed):
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full test set)
        self.min_top1_err = 100.0
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_samples = 0
        self.cur_iter = cur_iter
        self.filelogger = BufferedFileLogger(
            file_name=f'testing_metrics.csv',
            buffer_size=1,
            file_path=cfg.EXP_DIR,
            header=("al_budget", "epoch", "top1_err", "min_top1_err")
        )

    def reset(self, min_errs=False):
        if min_errs:
            self.min_top1_err = 100.0
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.num_top1_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, mb_size):
        self.mb_top1_err.add_value(top1_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = metrics.gpu_mem_usage()
        iter_stats = {
            '_type': 'test_iter',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'iter': '{}/{}'.format(cur_iter + 1, self.max_iter),
            'top1_err': self.mb_top1_err.get_win_median(),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        lu.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        top1_err = self.num_top1_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            '_type': 'test_epoch',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'top1_err': top1_err,
            'min_top1_err': self.min_top1_err
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        lu.log_json_stats(stats)
        self.filelogger.add_scalar(self.cur_iter,int(stats["epoch"].split('/')[0]),
                                   stats['top1_err'],
                                   stats['min_top1_err'])

    def close(self):
        self.filelogger.close()


class ValMeter(object):
    """Measures Validation stats."""

    def __init__(self, max_iter, cur_iter, seed):
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full Val set)
        self.min_top1_err = 100.0
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_samples = 0
        self.cur_iter=cur_iter
        self.filelogger = BufferedFileLogger(
            file_name=f'val_metrics.csv',
            buffer_size=200,
            file_path=cfg.EXP_DIR,
            header=("al_budget", "epoch", "top1_err", 'min_top1_err')
        )

    def reset(self, min_errs=False):
        if min_errs:
            self.min_top1_err = 100.0
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.num_top1_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, mb_size):
        self.mb_top1_err.add_value(top1_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = metrics.gpu_mem_usage()
        iter_stats = {
            '_type': 'Val_iter',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'iter': '{}/{}'.format(cur_iter + 1, self.max_iter),
            'top1_err': self.mb_top1_err.get_win_median(),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        lu.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        top1_err = self.num_top1_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            '_type': 'Val_epoch',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'top1_err': top1_err,
            'min_top1_err': self.min_top1_err
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        lu.log_json_stats(stats)
        self.filelogger.add_scalar(self.cur_iter, int(stats["epoch"].split('/')[0]),
                                   (stats['top1_err'],
                                                             stats['min_top1_err']))


    def close(self):
        self.filelogger.close()
