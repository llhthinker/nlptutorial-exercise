"""
A word segmentation program
"""
import argparse
import math


def load_model(model_file):
    """
    load n-gram language model
    """
    probs = dict()
    with open(model_file, 'r') as f:
        for line in f:
            word_and_prob = line.split('\t')
            probs[word_and_prob[0]] = float(word_and_prob[1])
    return probs


def get_prob(probs, word):
    """
    get word(n-gram) probability using linear smoothing  
    """
    lambda_1 = 0.95
    vocab_size = 1e6
    p = (1 - lambda_1) * (1 / vocab_size)
    if word in probs:
        p += lambda_1 * probs[word]
    return p


def segment(input_file, answer_file, probs):
    """
    word segmentation using probabilities(n-gram language model)
    and viterbi algroithm
    """
    best_edge = dict()
    best_score = dict()
    INF = 1e10
    seg_texts = []
    with open(input_file, 'r') as f:
        for line in f:
            # Forward step
            line = line.strip()
            best_edge[0] = None
            best_score[0] = 0
            for word_end in range(1, len(line) + 1):
                best_score[word_end] = INF
                for word_begin in range(word_end):
                    word = line[word_begin:word_end]
                    if word in probs or len(word) == 1:
                        p = get_prob(probs, word)
                        my_score = best_score[word_begin] + (-math.log2(p))
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = (word_begin, word_end)
            # Backward step
            words = []
            next_edge = best_edge[len(best_edge) - 1]

            while next_edge is not None:
                word = line[next_edge[0]:next_edge[1]]
                words.append(word)
                next_edge = best_edge[next_edge[0]]
            words.reverse()

            seg_line = " ".join(words)
            seg_texts.append(seg_line + '\n')

    with open(answer_file, 'w') as f:
        f.writelines(seg_texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--answer-file", type=str)
    args = parser.parse_args()

    probs = load_model(args.model_file)
    segment(args.input_file, args.answer_file, probs)
