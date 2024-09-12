import numpy as np

class AdaptiveCutoffThreshold:
    def __init__(self, patience=2, beta=0.01):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.beta = beta


    def __call__(self, metrics, num_classes, *args, **kwargs):
        if 'val_avg_acc' not in metrics:
            return

        score = metrics['val_avg_acc']

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta(num_classes):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


    def delta(self, num_classes):
        return self.beta / np.log(num_classes)


