from matplotlib import plt as plt
from tensorflow.keras.callbacks import Callback
"""Calculates a cutoff performance threshold, below which a model stops training
"""

class TerminateOnThreshold(Callback):
    '''Adaptive Cutoff Threshold (ACT)

    Keras Callback that terminates training if a given :code:`val_sparse_categorical_accuracy`
    dynamic threshold is not reached after ε epochs.
    The termination threshold has a logarithmic nature where the threshold
    increases by a decaying factor.

    Attributes:
        beta (float): threshold coefficient (captures the leniency of the calculated threshold)
        monitor (str): the optimizer metric type to monitor and calculate ACT on
        n_classes (int): number of classes
        zeta (float): diminishing factor; a positive, non-zero factor that controls how steeply the function horizontally asymptotes at y = 1.0 (i.e 100% accuracy)
    '''

    def __init__(self,
                monitor='val_sparse_categorical_accuracy',
                threshold_multiplier=0.25,
                diminishing_factor=0.25,
                n_classes = None):
        '''Initialize threshold-based termination callback

        Args:
            monitor (str, optional): the optimizer metric type to monitor and calculate ACT on
            threshold_multiplier (float, optional): threshold coefficient (captures the leniency of the calculated threshold)
            diminishing_factor (float, optional): iminishing factor; a positive, non-zero factor that controls how steeply the function horizontally asymptotes at y = 1.0 (i.e 100% accuracy)
            n_classes (None, optional): number of classes / output neurons
        '''

        super(TerminateOnThreshold, self).__init__()

        self.monitor = monitor
        self.beta = threshold_multiplier
        self.zeta = diminishing_factor
        self.n_classes = n_classes

    def get_threshold(self, epoch):
        '''Calculates the termination threshold given the current epoch

            ΔThreshold = ß(1 - (1 / n))
            Threshold_base = (1 / n) + ΔThreshold = (1 / n) + ß(1 - (1 / n))
                                                  = (1 + ßn - ß) / n
            Range of Threshold_base = (1 / n, 1) ; horizontal asymptote at 1
            ΔThreshold decays as the number of classes decreases

            --------------

            To account for the expected increase in accuracy over the number
            of epochs ε, a growth_factor is added to the base threshold:
            growth_factor = (1 - Threshold_base) - (1 / (1 / 1-Threshold_base) + ζ(ε - 1))

            Threshold_adaptive = Threshold_base + growth_factor
            Range of growth_factor = [Threshold_base, 1) ; horizontal asymptote at 1

        Args:
            epoch (int): current epoch

        Returns:
            float: calculated cutoff threshold
        '''

        baseline = 1.0 / self.n_classes     # baseline (random) val_acc
        complement_baseline = 1 - baseline
        delta_threshold = complement_baseline * self.beta
        base_threshold = baseline + delta_threshold
        ''' n_classes = 10, threshold_multiplier = 0.15 '''
        ''' yields .325 acc threshold for epoch 1 '''

        # epoch-based decaying increase in val_acc threshold
        complement_threshold = 1 - base_threshold    # the increase factor's upper limit
        growth_denom = (1.0 / complement_threshold) + self.zeta * (epoch - 1)
        growth_factor = complement_threshold - 1.0 / growth_denom

        calculated_threshold = base_threshold + growth_factor
        '''
            Same settings as before yields:
            epoch 1 = .325000
            epoch 2 = .422459,
            epoch 3 = .495327,
            epoch 4 = .551867,
            epoch 5 = .597014
        '''

        return calculated_threshold


    def on_epoch_end(self, epoch, logs=None):
        '''Called by Keras backend after each epoch during :code:`.fit()` & :code:`.evaluate()`

        Args:
            epoch (int): current epoch
            logs (None, optional): contains all the monitors (or metrics) used by the optimizer in the training and evaluation contexts
        '''

        logs = logs or {}

        if self.model is None:
            return

        if self.n_classes is None:
            self.n_classes = self.model.layers[-1].output_shape[1]

        threshold = self.get_threshold(epoch + 1)

        if self.monitor in logs:
            val_acc = logs[self.monitor]
            if val_acc < threshold:
                # threshold not met, terminate
                print(f'\nEpoch {(epoch + 1)}: Accuracy ({val_acc}) has not reached the baseline threshold {threshold}, terminating training... \n')
                self.model.stop_training = True



''' TESTING THRESHOLD EVAL '''
threshold_multiplier = 0.25
diminishing_factor = 0.25
n_classes = 10
n_epochs = 50

thresholds = []
for epoch in range(1, n_epochs + 1):

    baseline = 1.0 / n_classes     # baseline (random) val_acc
    complement_baseline = 1 - baseline
    delta_threshold = complement_baseline * threshold_multiplier
    base_threshold = baseline + delta_threshold

    # epoch-based decaying increase in val_acc threshold
    complement_threshold = 1 - base_threshold    # the increase factor's upper limit
    growth_denom = (1.0 / complement_threshold) + diminishing_factor * (epoch - 1)
    growth_factor = complement_threshold - 1.0 / growth_denom

    calculated_threshold = base_threshold + growth_factor

    thresholds.append(calculated_threshold)

plt.plot(thresholds, '.')
plt.show()

# print first 5 thresholds
print('\n\n')
[print(f'Epoch {t+1} threshold: {thresholds[t]}') for t in range(0, 5)];


