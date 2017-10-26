"""
Reads a bigram model and calculates entropy on the test set
Test train-bigram on test/02-train-input.txt
"""
import argparse
import math
from collections import defaultdict


def load_model(model_file):
    probs = defaultdict(int)
    with open(model_file, 'r') as f:
        for line in f:
            w_P = line.split('\t')
            probs[w_P[0]] = float(w_P[1])

    return probs


def test_bigram(test_file, probs):
    lambda_1 = 0.95
    lambda_2 = 0.95
    vocab_size = 1000000
    test_words_num, entropy_H = 0, 0

    with open(test_file, 'r') as f:
        for line in f:
            words = line.split()
            words.insert(0, "<s>")
            words.append("</s>")
            for i in range(1, len(words)):
                # Smoothed unigram probability
                p1 = lambda_1 * probs[words[i]] + (1 - lambda_1) / vocab_size
                # Smoothed bigram probability
                p2 = lambda_2 * probs[words[i - 1] +
                                      " " + words[i]] + (1 - lambda_2) * p1
                entropy_H += (-math.log2(p2))

                test_words_num += 1

    print("entropy = {}".format(entropy_H / test_words_num))


def test_ngram(test_file, probs, n):
    lambda_1 = 0.95
    vocab_size = 1000000
    test_words_num, entropy_H = 0, 0

    # Witten-Bell Smoothing
    unique_words = defaultdict(set)
    unique_counts = dict()
    counts = defaultdict(int)
    texts = []
    with open(test_file, 'r') as f:
        for line in f:
            texts.append(line)

            words = line.split()
            words.insert(0, "<s>")
            words.append("</s>")
            for i in range(n - 1, len(words)):
                for j in range(1, n):
                    # count W_i-n+1:i-1
                    counts[" ".join(words[i - j:i])] += 1
                    unique_words[" ".join(words[i - j:i])].add(words[i])

    for ngram in unique_words:
        unique_counts[ngram] = len(unique_words[ngram])

    for line in texts:
        words = line.split()
        words.insert(0, "<s>")
        words.append("</s>")
        for i in range(n - 1, len(words)):
            # Smoothed unigram probability
            p_j = lambda_1 * probs[words[i]] + (1 - lambda_1) / vocab_size
            for j in range(1, n):
                w_i_1 = " ".join(words[i - j:i])
                lambda_j = 1 - unique_counts[w_i_1] / \
                    (unique_counts[w_i_1] + counts[w_i_1])
                # Smoothed ngram (n=j+1) probability
                p_j = lambda_j * \
                    probs[" ".join(words[i - j:i + 1])] + (1 - lambda_j) * p_j

            entropy_H += (-math.log2(p_j))

            test_words_num += 1

    print("entropy = {}".format(entropy_H / test_words_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--test-file", type=str)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()

    probs = load_model(args.model_file)
    test_ngram(args.test_file, probs, args.n)
