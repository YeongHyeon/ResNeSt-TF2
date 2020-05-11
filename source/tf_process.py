import os, inspect

import tensorflow as tf
import numpy as np

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    iteration = 0

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            loss, accuracy, class_score = neuralnet.step(x=x_tr, y=y_tr, iteration=iteration, train=True)

            iteration += 1
            if(terminator): break

            neuralnet.save_params()
            
        print("Epoch [%d / %d] (%d iteration)  Loss:%.5f, Acc:%.5f" \
            %(epoch, epochs, iteration, loss, accuracy))

def test(neuralnet, dataset, batch_size):

    try: neuralnet.load_params()
    except: print("Parameter loading was failed")

    print("\nTest...")

    confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.
        loss, accuracy, class_score = neuralnet.step(x=x_te, y=y_te, train=False)

        label, logit = np.argmax(y_te[0]), np.argmax(class_score)
        confusion_matrix[label, logit] += 1

        if(terminator): break

    print("\nConfusion Matrix")
    print(confusion_matrix)

    tot_precision, tot_recall, tot_f1score = 0, 0, 0
    diagonal = 0
    for idx_c in range(dataset.num_class):
        precision = confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[:, idx_c])
        recall = confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[idx_c, :])
        f1socre = 2 * (precision * recall / (precision + recall))

        tot_precision += precision
        tot_recall += recall
        tot_f1score += f1socre
        diagonal += confusion_matrix[idx_c, idx_c]
        print("Class-%d | Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
            %(idx_c, precision, recall, f1socre))

    accuracy = diagonal / np.sum(confusion_matrix)
    print("\nTotal | Accuracy: %.5f, Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
        %(accuracy, tot_precision/dataset.num_class, tot_recall/dataset.num_class, tot_f1score/dataset.num_class))
