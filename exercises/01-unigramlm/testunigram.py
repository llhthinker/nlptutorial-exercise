"""Reads a unigram model and calculates
entropy and coverage for the test set"""

import math
import argparse


def load_model(model_file):
    probs = dict()
    with open(model_file, 'r') as f:
        for line in f:
            w_P = line.split('\t')
            probs[w_P[0]] = float(w_P[1])

    return probs


def test(test_file, probs):
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    vocab_size = 1000000
    test_words_num, entropy_H = 0, 0
    unk_num = 0

    with open(test_file, 'r') as f:
        for line in f:
            words = line.split()
            words.append("</s>")  # the end of words
            for w in words:
                test_words_num += 1
                word_prob = lambda_unk / vocab_size
                if w in probs:
                    word_prob += lambda_1 * probs[w]
                else:
                    unk_num += 1
                entropy_H += (-math.log2(word_prob))

    print("entropy = {}".format(entropy_H / test_words_num))
    print("coverage = {}".format((test_words_num - unk_num) / test_words_num))


if __name__ == "__main__":
	"""
	python testunigram.py  --test-file ../../test/01-train-input.txt
	python testunigram.py  --test-file ../../test/01-test-input.txt
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument("--test-file", type=str)
	args = parser.parse_args()

	model_file = "../../model/wiki-en-train.word.unigram.model"
	probs = load_model(model_file)

	test(args.test_file, probs)
