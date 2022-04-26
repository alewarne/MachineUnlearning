from scipy.optimize import minimize
from Unlearner.LRUnlearner import LogisticRegressionUnlearner
from scipy.special import expit
import numpy as np
import scipy.sparse as sp
import time
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
from scipy.linalg import inv
import json


class DPLRUnlearner(LogisticRegressionUnlearner):
    def __init__(self, train_data, test_data, voc, epsilon, delta, sigma, lambda_, b=None):
        self.set_train_test_data(train_data, test_data)
        self.lambda_ = lambda_
        self.feature2dim = voc
        self.dim2feature = {v:k for k,v in self.feature2dim.items()}
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = sigma
        self.lambda_ = lambda_
        self.theta = np.random.standard_normal(self.x_train.shape[1])  # sample weights normal distributed
        self.model_param_str = 'lambda={}_epsilon={}_delta={}_sigma={}'.format(
            self.lambda_, self.epsilon, self.delta, self.sigma)
        if b is None:
            self.b = np.random.normal(0, self.sigma, size=self.x_train.shape[1]) if self.sigma != 0 else np.zeros(self.x_train.shape[1])
        else:
            self.b = b
        self.gradient_calls = 0

    # computes l(x,y;theta). if x and y contain multiple samples l is summed up over them
    # we use l(x,y) = log(1+exp(-y*theta^T*x))
    def get_loss_l(self, theta, x, y):
        dot_prod = x.dot(theta) * y
        data_loss = -np.log(expit(dot_prod))
        total_loss = np.sum(data_loss, axis=0)
        return total_loss

    # computes L(x,y;theta)
    def get_loss_L(self, theta, x, y):
        summed_loss = self.get_loss_l(theta, x, y)
        total_loss = summed_loss + 0.5*self.lambda_*np.dot(theta, theta.T) + np.dot(self.b, theta)
        return total_loss

    # return total loss L on train set.
    def get_train_set_loss(self, theta):
        return self.get_loss_L(theta, self.x_train, self.y_train)

    # get gradient w.r.t. parameters (-y*x*sigma(-y*Theta^Tx)) for y in {-1,1}
    def get_gradient_l(self, theta, x, y):
        assert x.shape[0] == y.shape[0], f'{x.shape[0]} != {y.shape[0]}'
        dot_prod = x.dot(theta) * y
        factor = -expit(-dot_prod) * y
        # we need to multiply every row of x by the corresponding value in factor vector
        if type(x) is sp.csr_matrix:
            factor_m = sp.diags(factor)
            res = factor_m.dot(x)
        else:
            res = np.expand_dims(factor, axis=1) * x
        grad = res.sum(axis=0)
        if type(grad) is np.matrix:
            grad = grad.A
        return grad

    def get_gradient_L(self, theta, x, y):
        summed_grad = self.get_gradient_l(theta, x, y)
        total_grad = summed_grad + self.lambda_ * theta + self.b
        total_grad = total_grad.squeeze()
        self.gradient_calls += 1
        return total_grad

    # this is the gradient of L on the train set. This should be close to zero after fitting.
    def get_train_set_gradient(self, theta):
        return self.get_gradient_L(theta, self.x_train, self.y_train)

    # computes inverse hessian for data x. As we only need the inverse hessian on the entire dataset we return the
    # Hessian on the full L loss.
    def get_inverse_hessian(self, x):
        dot = x.dot(self.theta)
        probs = expit(dot)
        weighting = probs * (1-probs) # sigma(-t) = (1-sigma(t))
        if type(x) is sp.csr_matrix:
            weighting_m = sp.diags(weighting)
            p1 = x.transpose().dot(weighting_m)
        else:
            p1 = x.transpose() * np.expand_dims(weighting, axis=0)
        res = p1.dot(x)
        res += self.lambda_ * np.eye(self.dim)  # hessian of regularization
        cov_inv = inv(res)
        return cov_inv

    def get_first_order_update(self, G, unlearning_rate):
        return self.theta - unlearning_rate * G

    def get_second_order_update(self, x, y, G):
        H_inv = self.get_inverse_hessian(x)
        return self.theta - np.dot(H_inv, G)

    def get_fine_tuning_update(self, x, y, learning_rate, batch_size=32):
        new_theta = self.theta.copy()
        for i in range(0, x.shape[0], batch_size):
            grad = self.get_gradient_L(new_theta, x[i:i+batch_size], y[i:i+batch_size])
            new_theta -= 1./batch_size * learning_rate * grad
        return new_theta

    # given indices_to_delete (i.e. column indices) computes row indices where the column indices are non-zero
    def get_relevant_indices(self, indices_to_delete):
        # get the rows (samples) where the features appear
        relevant_indices = self.x_train[:, indices_to_delete].nonzero()[0]
        # to avoid having samples more than once
        relevant_indices = np.unique(relevant_indices)
        return relevant_indices

    def get_G(self, z, z_delta):
        """
        Computes G as defined in the paper using z=(x,y) and z_delta=(x_delta, y_delta)
        :param z: Tuple of original (unchanged) data (np.array /csr_matrix , np.array)
        :param z_delta: Tuple of changed data (np.array /csr_matrix , np.array)
        :return: G=\sum \nabla l(z_delta)-\nabla l(z)
        """
        grad_z_delta = self.get_gradient_l(self.theta, z_delta[0], z_delta[1])
        grad_z = self.get_gradient_l(self.theta, z[0], z[1])
        diff = grad_z_delta - grad_z
        if type(z[0]) is sp.csr_matrix:
            diff = diff.squeeze()
        return diff

    def predict(self, x, theta):
        logits = expit(x.dot(theta))
        y_pred = np.array([1 if l >= 0.5 else -1 for l in logits])
        return y_pred

    def get_performance(self, x, y, theta):
        assert x.shape[0] == y.shape[0], '{} != {}'.format(x.shape[0], y.shape[0])
        logits = expit(x.dot(theta))
        y_pred = np.array([1 if l >= 0.5 else -1 for l in logits])
        accuracy = len(np.where(y_pred == y)[0])/x.shape[0]
        fpr, tpr, _ = roc_curve(y, logits)
        prec, rec, _ = precision_recall_curve(y, logits)
        auc_roc = auc(fpr, tpr)
        auc_pr = auc(rec, prec)
        report = classification_report(y, y_pred, digits=4, output_dict=True)
        n_data = x.shape[0]
        loss = self.get_loss_L(theta, x, y)
        grad = self.get_gradient_L(theta, x, y)
        report['test_loss'] = loss
        report['gradient_norm'] = np.sum(grad**2)
        report['train_loss'] = self.get_train_set_loss(theta)
        report['gradient_norm_train'] = np.sum(self.get_train_set_gradient(theta)**2)
        report['accuracy'] = accuracy
        report['test_roc_auc'] = auc_roc
        report['test_pr_auc'] = auc_pr
        return report

    def fit_model(self):
        start_time = time.time()
        #res = minimize(self.get_train_set_loss, self.theta, method='L-BFGS-B', jac=self.get_train_set_gradient,
        #               options={'disp':True})
        res = minimize(self.get_train_set_loss, self.theta, method='L-BFGS-B', jac=self.get_train_set_gradient,
                       options={'maxiter': 1000})
        end_time = time.time()
        total_time = end_time-start_time
        self.theta = res.x
        #print(f'Fitting took {total_time} seconds.')
        performance = self.get_performance(self.x_test, self.y_test, self.theta)
        acc = performance['accuracy']
        gr = performance['gradient_norm_train']
        print(f'Achieved accuracy: {acc}')
        print(f'Gradient residual train: {gr}')
        #print(json.dumps(performance, indent=4))

    def get_n_largest_features(self, n):
        theta_abs = np.abs(self.theta)
        largest_features_ind = np.argsort(-theta_abs)[:n]
        largest_features = [self.dim2feature[d] for d in largest_features_ind]
        return largest_features_ind, largest_features
