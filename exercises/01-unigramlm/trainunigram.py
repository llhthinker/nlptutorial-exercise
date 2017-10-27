"""Creates a unigram model"""
import argparse

def train(training_file, model_file):
    counts = dict()
    total_count = 0
    with open(training_file, 'r') as f:
        for line in f:
            words = line.split()
            words.append("</s>")  # the end of words
            for w in words:
                if w not in counts:
                    counts[w] = 1
                else:
                    counts[w] += 1
                total_count += 1

    with open(model_file, 'w') as f:
        sorted_counts = sorted(
            counts.items(), key=lambda d: d[1],	reverse=True)
        for word, count in sorted_counts:
            probability = float(counts[word]) / total_count
            f.write(word + "\t" + str(probability) + "\n")


if __name__ == "__main__":
    # training_file = "../../data/wiki-en-train.word"
    # model_file = "../../model/wiki-en-train.word.unigram.model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-file", type=str)
    parser.add_argument("--model-file", type=str)
    args = parser.parse_args()

    print("train...")
    train(args.training_file, args.model_file)
    print("train over")
