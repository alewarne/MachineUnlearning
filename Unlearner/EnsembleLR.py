from .ensemble import softmax
from sklearn.metrics import classification_report
import numpy as np


class LinearEnsemble:

    def __init__(self, models, n_classes):
        #assert len(models) == len(training_data_splits)
        self.models = models
        self.n_classes = n_classes

    def predict(self, x):
        preds = np.zeros((x.shape[0], len(self.models)), dtype=np.int64)
        for i, model_tuple in enumerate(self.models):
            model = model_tuple[0]
            preds[:, i] = model.predict(x, model.theta)
        preds[np.where(preds == -1)] = 0
        preds = np.apply_along_axis(np.bincount, axis=1, arr=preds, minlength=self.n_classes)
        preds_max = np.argmax(preds, axis=1)
        preds_max[np.where(preds_max == 0)[0]] = -1
        #probas = softmax(preds, axis=1)
        return preds_max

    def train_ensemble(self):
        for model_tuple in self.models:
            model_tuple[0].fit_model()

    def evaluate(self, x, y):
        Y_pred = self.predict(x)
        rep = classification_report(y, Y_pred, output_dict=True)
        return rep, rep['accuracy']

    def update_models(self, data_indices_to_delete):
        for i in range(len(self.models)):
            model, data_indices = self.models[i][0], self.models[i][1]
            to_delete = []
            for j, data_idx in enumerate(data_indices):
                if data_idx in data_indices_to_delete:
                    to_delete.append(j)
            new_model_indices = [k for k in range(model.x_train.shape[0]) if k not in to_delete]
            self.models[i][0].x_train = self.models[i][0].x_train[new_model_indices]
            self.models[i][0].y_train = self.models[i][0].y_train[new_model_indices]

    def get_gradient_calls(self):
        n_gradients = 0
        for model_tup in self.models:
            n_gradients += model_tup[0].gradient_calls * model_tup[0].x_train.shape[0]
        return n_gradients
