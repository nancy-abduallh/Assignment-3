import os
import time
from collections import OrderedDict, deque
from datetime import timedelta

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler

from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import lib.utils.misc as misc


class LTRTrainer(BaseTrainer):

    TARGET_SAMPLES = 2254
    _SPS_HISTORY_SIZE = 6  # number of recent interval sps values to average
    _MIN_INTERVAL_SEC = 0.25  # ignore intervals shorter than this when updating history

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()

        # Initialize statistics per loader
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # TensorBoard writer
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            os.makedirs(tensorboard_writer_dir, exist_ok=True)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # Timing and counters
        self.samples_processed = 0
        self.last_interval_start_time = None
        self.epoch_start_time = None
        self.last_reported_samples = 0
        self.sps_history = deque(maxlen=self._SPS_HISTORY_SIZE)

    def _set_default_settings(self):
        defaults = {'print_interval': 100, 'print_stats': None, 'description': ''}
        for param, val in defaults.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, val)

    def cycle_dataset(self, loader):
        """Run one pass over the provided loader but stop after TARGET_SAMPLES."""
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()
        self.samples_processed = 0
        self.last_reported_samples = 0
        self.sps_history.clear()

        # allow override via settings (keeps config untouched)
        total_samples = getattr(self.settings, 'train_stop_samples', None) or self.TARGET_SAMPLES

        for i, data in enumerate(loader, 1):
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # Compute IoU-based accuracy
            iou_value = stats.get("IoU", 0)
            if isinstance(iou_value, torch.Tensor):
                acc = (iou_value.mean().item() if iou_value.numel() > 0 else 0.0)
            else:
                acc = float(iou_value)
            acc = max(min(acc * 100.0, 100.0), 0.0)
            stats["Accuracy"] = acc

            # Backprop
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Accurate batch size (handles variable last batch sizes)
            try:
                batch_size = int(data['template_images'].shape[loader.stack_dim])
            except Exception:
                batch_size = getattr(self.settings, 'BATCH_SIZE', 4)

            # Update stats and counters
            self._update_stats(stats, batch_size, loader)
            self.samples_processed += batch_size

            # When printing, compute actual samples and time for interval
            if (self.samples_processed - self.last_reported_samples) >= self.settings.print_interval or self.samples_processed >= total_samples or i == len(loader):
                self._print_stats(i, loader, batch_size, total_samples)

            # Enforce hard stop at total_samples
            if self.samples_processed >= total_samples:
                break

        # CRITICAL FIX: Step LR scheduler at the END of epoch, not beginning
        if loader.training and self.lr_scheduler is not None:
            # note: last_epoch should already be restored from checkpoint -> step uses correct counters
            try:
                self.lr_scheduler.step()
            except Exception as e:
                print(f"⚠️ LR scheduler step failed at epoch {self.epoch}: {e}")

        # End of epoch summary
        self._summarize_epoch(loader)

    def train_epoch(self):
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.last_interval_start_time = self.start_time
        self.epoch_start_time = self.start_time
        self.last_reported_samples = 0
        self.sps_history.clear()

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name]:
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size, total_samples):
        current_time = time.time()
        time_last_interval = current_time - self.last_interval_start_time
        time_since_beginning = current_time - self.epoch_start_time

        # Number of samples processed since last report (handles variable batch sizes)
        interval_samples = self.samples_processed - self.last_reported_samples
        if interval_samples <= 0:
            interval_samples = max(1, batch_size)

        # Instantaneous samples/sec for the last interval
        interval_sps = interval_samples / time_last_interval if time_last_interval > 0 else 0.0

        # Only update history if interval duration is reasonable (avoids noisy short intervals)
        if time_last_interval >= self._MIN_INTERVAL_SEC and interval_sps > 0:
            self.sps_history.append(interval_sps)

        # Compute smoothed samples/sec as mean of history (fallback to cumulative rate)
        if len(self.sps_history) > 0:
            mean_sps = float(sum(self.sps_history) / len(self.sps_history))
        else:
            mean_sps = (self.samples_processed / time_since_beginning) if time_since_beginning > 0 else 0.0

        samples_per_second = mean_sps if mean_sps > 0 else (self.samples_processed / time_since_beginning if time_since_beginning > 0 else 0)

        samples_remaining = max(total_samples - self.samples_processed, 0)
        time_remaining = samples_remaining / samples_per_second if samples_per_second > 0 else 0

        time_last_interval_str = str(timedelta(seconds=int(time_last_interval)))
        time_since_beginning_str = str(timedelta(seconds=int(time_since_beginning)))
        time_remaining_str = str(timedelta(seconds=int(time_remaining)))

        loss_val = self.stats[loader.name].get('Loss/total', AverageMeter()).avg
        iou_val = self.stats[loader.name].get('IoU', AverageMeter()).avg
        acc_val = self.stats[loader.name].get('Accuracy', AverageMeter()).avg

        print_str = (
            f"Epoch {self.epoch} : {self.samples_processed} / {total_samples} samples , "
            f"time for last {interval_samples} samples : {time_last_interval_str} , "
            f"time since beginning : {time_since_beginning_str} , "
            f"time left to finish epoch : {time_remaining_str} , "
            f"Loss/total: {loss_val:.5f}, "
            f"IoU: {iou_val:.5f}, "
            f"Accuracy: {acc_val:.2f}%"
        )

        print(print_str)

        if misc.is_main_process():
            with open(self.settings.log_file, 'a') as f:
                f.write(print_str + '\n')

        # update interval trackers
        self.last_interval_start_time = current_time
        self.last_reported_samples = self.samples_processed
        self.prev_time = current_time

    def _summarize_epoch(self, loader):
        epoch_time = time.time() - self.epoch_start_time
        epoch_time_str = str(timedelta(seconds=int(epoch_time)))

        loss_avg = self.stats[loader.name].get('Loss/total', AverageMeter()).avg
        iou_avg = self.stats[loader.name].get('IoU', AverageMeter()).avg
        acc_avg = self.stats[loader.name].get('Accuracy', AverageMeter()).avg

        summary_str = (
            "\n" + "=" * 80 + "\n"
            f"✅ Epoch {self.epoch} Summary ({loader.name}):\n"
            f"Total samples processed: {self.samples_processed}\n"
            f"Total epoch time: {epoch_time_str}\n"
            f"Average Loss: {loss_avg:.5f}\n"
            f"Average IoU: {iou_avg:.5f}\n"
            f"Average Accuracy: {acc_avg:.2f}%\n"
            + "=" * 80 + "\n"
        )

        print(summary_str)
        if misc.is_main_process():
            with open(self.settings.log_file, 'a') as f:
                f.write(summary_str + '\n')

    def _stats_new_epoch(self):
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = f'LearningRate/group{i}'
                    if var_name not in self.stats[loader.name]:
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)
        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
