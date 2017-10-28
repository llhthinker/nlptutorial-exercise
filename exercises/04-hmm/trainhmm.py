"""
Train HMM to get the transition probabilities
and the emission probabilities
"""
import argparse
from collections import defaultdict


def train(training_file, model_file):
    # Input data format is "natural_JJ language_NN..."
    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)

    with open(training_file, 'r') as f:
        for line in f:
            previous = "<s>"    # <s> tag, Make the sentence start
            context[previous] += 1
            word_tags = line.split()
            for wt in word_tags:
                word, tag = tuple(wt.split('_'))
                emit[tag + " " + word] += 1
                transition[previous + " " + tag] += 1
                context[tag] += 1
                previous = tag
            # </s> tag, the end of sentence
            transition[previous + " </s>"] += 1

    with open(model_file, 'w') as f:
        # Write the transition probabilities
        for key, value in transition.items():
            previous = key.split()[0]
            # print("T", key, value / context[previous])
            f.write("T " + key + " " + str(value / context[previous]) + "\n")
        # Write the emission probabilities
        for key, value in emit.items():
            tag = key.split()[0]
            # print("E", key, value / context[tag])
            f.write("E " + key + " " + str(value / context[tag]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-file", type=str)
    parser.add_argument("--model-file", type=str)
    args = parser.parse_args()

    train(args.training_file, args.model_file)
