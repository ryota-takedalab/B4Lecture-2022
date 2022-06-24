import pickle
import argparse
import time

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from my_functions import hmm


def main():
    parser = argparse.ArgumentParser(description='ex8')
    parser.add_argument("-i", "--input", help="input file")
    args = parser.parse_args()
    data = pickle.load(open(f"../data{args.input}.pickle", "rb"))
    # answer model id, axis=(trial)
    answer_models = np.array(data["answer_models"])
    # output, axis=(trial, time)
    output = np.array(data["output"])
    # initial state, axis=(model, state, 1)
    pi = np.array(data["models"]["PI"])
    # transition probability, axis=(model, before, after)
    a = np.array(data["models"]["A"])
    # output probability, axis=(model, state, output)
    b = np.array(data["models"]["B"])

    # print(answer_models.shape)
    # print(output.shape)
    # print(pi.shape)
    # print(a.shape)
    # print(b.shape)
    # print(answer_models)
    # print(output)
    # print(pi)
    # print(a)
    # print(b)
    
    n_models, n_states, n_outputs = b.shape
    n_series = len(output)
    hmms = []
    forward_likelihoods = np.zeros((n_models, n_series))
    viterbi_likelihoods = np.zeros((n_models, n_series))
    
    forward_time = np.zeros(n_models)
    viterbi_time = np.zeros(n_models)
    for i in range(n_models):
        # create model
        hmms.append(hmm.HMM(n_states, pi[i].reshape((n_states)), a[i], b[i]))
        start_time = time.time()
        for j in range(n_series):
            # forward algorithm
            forward_likelihoods[i, j] = hmms[i].forward(output[j])
        middle_time = time.time()
        for j in range(n_series):
            viterbi_likelihoods[i, j] = hmms[i].viterbi(output[j])
        end_time = time.time()
        
        # process time
        forward_time[i] = middle_time - start_time
        viterbi_time[i] = end_time - middle_time
    forward_total_time = np.sum(forward_time)
    viterbi_total_time = np.sum(viterbi_time)
    print(f"total time(forward) :{forward_total_time}")
    print(f"total time(viterbi) :{viterbi_total_time}")
    forward_expected_models = np.argmax(forward_likelihoods, axis=0)
    viterbi_expected_models = np.argmax(viterbi_likelihoods, axis=0)

    # plot result
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    labels = [i for i in range(n_models)]
    column_labels = [f"pred_{i}" for i in labels]
    row_labels = [f"ans_{i}" for i in labels]

    # forward
    forward_accuracy = accuracy_score(answer_models, forward_expected_models)
    ax[0].set_title(
        f"forward on data{args.input}\n"
        f"acc:{forward_accuracy} time:{forward_total_time:.4f}s")
    forward_cm = confusion_matrix(answer_models, forward_expected_models,
                                  labels=labels)
    ax[0].pcolor(forward_cm, cmap=plt.cm.Blues)
    for (i, j), z in np.ndenumerate(forward_cm):
        ax[0].text(j + 0.5, i + 0.5, '{}'.format(z),
                   ha='center', va='center', color="red")

    ax[0].set_xticks(np.arange(forward_cm.shape[0]) + 0.5, minor=False)
    ax[0].set_yticks(np.arange(forward_cm.shape[1]) + 0.5, minor=False)

    ax[0].invert_yaxis()
    ax[0].xaxis.tick_top()

    ax[0].set_xticklabels(row_labels, minor=False)
    ax[0].set_yticklabels(column_labels, minor=False)

    # viterbi
    viterbi_accuracy = accuracy_score(answer_models, viterbi_expected_models)
    ax[1].set_title(
        f"viterbi on data{args.input}\n"
        f"acc={viterbi_accuracy} time:{viterbi_total_time:.4f}s")
    viterbi_cm = confusion_matrix(answer_models, viterbi_expected_models,
                                  labels=labels) 
    ax[1].pcolor(viterbi_cm, cmap=plt.cm.Blues)
    for (i, j), z in np.ndenumerate(viterbi_cm):
        ax[1].text(j + 0.5, i + 0.5, '{}'.format(z),
                   ha='center', va='center', color="red")

    ax[1].set_xticks(np.arange(viterbi_cm.shape[0]) + 0.5, minor=False)
    ax[1].set_yticks(np.arange(viterbi_cm.shape[1]) + 0.5, minor=False)

    ax[1].invert_yaxis()
    ax[1].xaxis.tick_top()

    ax[1].set_xticklabels(row_labels, minor=False)
    ax[1].set_yticklabels(column_labels, minor=False)
    plt.savefig(f"result_{args.input}.png")


if __name__ == "__main__":
    main()
