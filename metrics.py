import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, metric_name):
        self._name = metric_name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, **kwargs):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def name(self):
        return self._name

    def value(self):
        return self.avg


class F1Meter:
    def __init__(self, k, average='macro'):
        self._override_name = f'f1_score_{average}'
        self._average = average
        self.meter = AverageMeter(self._override_name)
        self.labels = range(k)

    def reset(self):
        self.meter.reset()

    def update(self, input, target, **kwargs):
        input = input.argmax(dim=1).cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        score = f1_score(target, input, labels=np.unique(target), average=self._average)
        self.meter.update(score, len(input))

    def value(self):
        return self.meter.avg

    @property
    def name(self):
        return self._override_name


class AccuracyMeter:
    def __init__(self):
        self._override_name = 'accuracy'
        self.meter = AverageMeter(self._override_name)

    def reset(self):
        self.meter.reset()

    def update(self, input, target, **kwargs):
        input = input.argmax(dim=1).cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        self.meter.update(accuracy_score(target, input), len(input))

    def value(self):
        return self.meter.avg

    @property
    def name(self):
        return self._override_name
