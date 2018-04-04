import tensorflow as tf
from tensorflow.python.estimator.inputs.queues import feeding_functions

class FastPredict:

    def _createGenerator(self):
        while self.next_features is not None:
            out = self.next_features
            self.next_features = None
            yield out


    def __init__(self, estimator):
        self.estimator = estimator
        self.first_run = True
        self.batch_size = 1
        self.gen = None

    def _input_fn(self):
        if self.gen is None:
            self.gen = self._createGenerator()

        return next(self.gen), None

    def predict(self, features):
        self.next_features = features
        if self.first_run:
            self.predictions = self.estimator.predict(
                input_fn=self._input_fn)
            self.first_run = False

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results
