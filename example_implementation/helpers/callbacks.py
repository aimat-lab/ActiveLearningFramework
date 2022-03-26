import os

import numpy as np
from keras.callbacks import Callback

from example_implementation.helpers import properties


class CallbackStopIfLossLow(Callback):

    def __init__(self, min_epoch, thr):
        super().__init__()
        self.thr, self.min_epoch = thr, min_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('loss') <= self.thr and epoch >= self.min_epoch:
            self.model.stop_training = True


class CallbackDocumentation(Callback):

    def __init__(self, entity):
        super().__init__()
        self._entity = entity

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filename = os.path.abspath(os.path.abspath(properties.results_location["loss_over_epochs"]))
        os.makedirs(filename, exist_ok=True)
        loss_history = np.load(os.path.join(filename, self._entity + properties.loss_history_suffix))
        last_loss = logs.get('loss')
        loss_history = np.append(loss_history, last_loss)
        np.save(os.path.join(filename, self._entity + properties.loss_history_suffix), np.asarray(loss_history))
