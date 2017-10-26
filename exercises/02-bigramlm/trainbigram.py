"""
Creates a bigram model
Train the model on data/wiki-en-train.word
"""
import argparse

from collections import defaultdict

def train_bigram(training_file, model_file):
    counts = defaultdict(int)
    context_counts = defaultdict(int)

    with open(training_file, 'r') as f:
        for line in f:
            words = line.split()
            words.insert(0, "<s>")  # add beginning label
            words.append("</s>")    # add end label
            words_len = len(words)
            for i in range(1, words_len):
                # Add bigram and bigram context
                bigram = words[i-1] + " " + words[i]
                counts[bigram] += 1
                context_counts[words[i-1]] += 1
                # Add unigram and unigram context
                unigram = words[i]
                counts[unigram] += 1
                context_counts[""] += 1     # unigram context is ""

    ngram_model = dict()
    for ngram in counts:
        words = ngram.split()
        words.pop()
        context = " ".join(words)
        prob = counts[ngram] / context_counts[context]
        ngram_model[ngram] = prob
    ngram_model = sorted(ngram_model.items(), key=lambda d: d[1], reverse=True)
    
    
    with open(model_file, 'w') as f:
        for ngram, prob in ngram_model:
            f.write(ngram+"\t"+str(prob)+"\n")


def train_ngram(training_file, model_file, n):
    counts = defaultdict(int)
    context_counts = defaultdict(int)

    with open(training_file, 'r') as f:
        for line in f:
            words = line.split()
            words.insert(0, "<s>")  # add beginning label
            words.append("</s>")    # add end label
            words_len = len(words)
            for i in range(n-1, words_len):
                for j in range(n):
                    # W_i-n+1,..., W_i
                    ngram = " ".join(words[i-n+1:i+1-j])
                    # W_i-n+1,..., W_i-1
                    n_1_gram = " ".join(words[i-n+1:i-j])
                    counts[ngram] += 1
                    context_counts[n_1_gram] += 1

    ngram_model = dict()
    for ngram in counts:
        words = ngram.split()
        words.pop()
        context = " ".join(words)
        prob = counts[ngram] / context_counts[context]
        ngram_model[ngram] = prob
    ngram_model = sorted(ngram_model.items(), key=lambda d: d[1], reverse=True)
    
    
    with open(model_file, 'w') as f:
        for ngram, prob in ngram_model:
            f.write(ngram+"\t"+str(prob)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-file", type=str)
    parser.add_argument("--model-file", type=str)
    # n-gram
    parser.add_argument("--n", type=int)
    
    args = parser.parse_args()

    train_ngram(args.training_file, args.model_file, args.n)

    

