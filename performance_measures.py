from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit

from colorama import Fore
from sklearn.metrics import auc, roc_curve, precision_score, recall_score

from utils.vocab import Vocabulary
from utils.reader import Data
from utils.utils import print_progress, create_checkpoints_dir

params = {
    "batch_size": 128,
    "embed_size": 64,
    "hidden_size": 64,
    "num_layers": 2,
    "checkpoints": "./checkpoints/",
    "std_factor": 6.,
    "dropout": 0.7,
}

path_normal_data = "datasets/vulnbank_train.txt"
path_anomaly_data = "datasets/vulnbank_anomaly.txt"

# create_checkpoints_dir(params["checkpoints"])

vocab = Vocabulary()
params["vocab"] = vocab

d = Data(path_normal_data)


class Predictor():
    def __init__(self, checkpoints_path, std_factor, vocab):

        self.threshold = 0.
        self.checkpoints = checkpoints_path
        self.path_to_graph = checkpoints_path + 'seq2seq'
        self.std_factor = std_factor
        self.vocab = vocab
        self.__load()

    def __load(self):
        """
        Loads model from the checkpoint directory and sets models params.
        """
        try:
            loaded_graph = tf.Graph()
            with loaded_graph.as_default():
                saver = tf.train.import_meta_graph(
                    self.path_to_graph + '.meta')

            self.sess = tf.Session(graph=loaded_graph)
            saver.restore(self.sess, tf.train.latest_checkpoint(
                self.checkpoints))

            # loading model parameters
            self.inputs = loaded_graph.get_tensor_by_name('inputs:0')
            self.targets = loaded_graph.get_tensor_by_name('targets:0')
            self.lengths = loaded_graph.get_tensor_by_name('lengths:0')
            self.dropout = loaded_graph.get_tensor_by_name('dropout:0')
            self.batch_size_tensor = loaded_graph.get_tensor_by_name('batch_size:0')
            self.seq_len_tensor = loaded_graph.get_tensor_by_name('max_seq_len:0')
            self.get_batch_loss = loaded_graph.get_tensor_by_name('batch_loss:0')
            self.get_probabilities = loaded_graph.get_tensor_by_name('probs:0')
            self.get_logits = loaded_graph.get_tensor_by_name('logits:0')

        except Exception as e:
            raise ValueError('Unable to create model: {}'.format(e))

    def set_threshold(self, data_gen):
        """
        Calculates threshold for anomaly detection.
        """

        total_loss = []
        for seq, l in data_gen:
            batch_loss, _ = self._predict_for_request(seq, l)
            total_loss.extend(batch_loss)

        mean = np.mean(total_loss)
        std = np.std(total_loss)
        self.threshold = mean + self.std_factor * std

        print('Validation loss mean: ', mean)
        print('Validation loss std: ', std)
        print('Threshold for anomaly detection: ', self.threshold)

        return self.threshold

    def predict(self, data_gen, visual=True):
        """
        Predicts probabilities and loss for given sequences.
        """
        loss = []
        predictions = []
        num_displayed = 0

        for seq, l in data_gen:
            batch_loss, alphas = self._predict_for_request(seq, l)
            loss.extend(batch_loss)
            alphas = self._process_alphas(seq, alphas, 1)
            mask = np.array([l > self.threshold for l in batch_loss])
            final_pred = mask.astype(int)
            predictions.extend(final_pred)

            if visual and num_displayed < 10 and final_pred == [1]:
                print('\n\nPrediction: ', final_pred[0])
                print('Loss ', batch_loss[0])

                num_displayed += 1
                self._visual(alphas, seq)

        return predictions, loss

    def _predict_for_request(self, X, l):
        """
        Predicts probabilities and loss for given data.
        """
        lengths = [l]
        max_seq_len = l
        feed_dict = {
            self.inputs: X,
            self.targets: X,
            self.lengths: lengths,
            self.dropout: 1.0,
            self.batch_size_tensor: 1,
            self.seq_len_tensor: max_seq_len}

        fetches = [self.get_batch_loss, self.get_probabilities]
        batch_loss, alphas = self.sess.run(fetches, feed_dict=feed_dict)

        return batch_loss, alphas

    def _process_alphas(self, X, alphas, batch_size):
        """
        Counts numbers as probabilities for given data sample.
        """
        processed_alphas = []
        for i in range(batch_size):
            probs = alphas[i]
            coefs = np.array([probs[j][X[i][j]] for j in range(len(X[i]))])
            coefs = coefs / coefs.max()
            processed_alphas.append(coefs)

        return processed_alphas

    def _visual(self, alphas, X):
        """
        Colors sequence of malicious characters.
        """
        for i, x in enumerate(X):
            coefs = alphas[i]
            tokens = self.vocab.int_to_string(x)

            for j in range(len(x)):
                token = tokens[j]
                if coefs[j] < 0.09:
                    c = Fore.GREEN
                else:
                    c = Fore.BLACK
                if token != '<PAD>' and token != '<EOS>':
                    token = ''.join(c + token)
                    print(token, end='')

            print(Fore.BLACK + '', end='')


p = Predictor(params["checkpoints"], params["std_factor"], params["vocab"])

val_gen = d.val_generator()
threshold = p.set_threshold(val_gen)

# Benign samples

test_gen = d.test_generator()
valid_preds, valid_loss = p.predict(test_gen)

print('Number of FP: ', np.sum(valid_preds))
print('Number of samples: ', len(valid_preds))
print('FP rate: {:.4f}'.format(np.sum(valid_preds) / len(valid_preds)))

# Anomalous samples

pred_data = Data(path_anomaly_data, predict=True)
pred_gen = pred_data.predict_generator()
anomaly_preds, anomaly_loss = p.predict(pred_gen)

print('Number of TP: ', np.sum(anomaly_preds))
print('Number of samples: ', len(anomaly_preds))
print('TP rate: {:.4f}'.format(np.sum(anomaly_preds) / len(anomaly_preds)))

# Testing
y_true = np.concatenate(([0] * len(valid_preds), [1] * len(anomaly_preds)), axis=0)
preds = np.concatenate((valid_preds, anomaly_preds), axis=0)
loss_pred = np.concatenate((valid_loss, anomaly_loss), axis=0)
assert len(y_true) == len(loss_pred)

precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))

fpr, tpr, _ = roc_curve(y_true, loss_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
