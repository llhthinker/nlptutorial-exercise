import argparse
import pickle

from trainperceptron import predict_one, create_features


def predict_all(model_file, input_file, answer_file):
    with open(model_file, 'rb') as f:
        weight = pickle.load(f)
    answers = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            phi = create_features(line)
            predicted_y = predict_one(weight, phi)
            answers.append(str(predicted_y) + "\t" + line + "\n")

    with open(answer_file, 'w') as f:
        f.writelines(answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--answer-file", type=str)
    args = parser.parse_args()
    print("predicting")
    predict_all(args.model_file, args.input_file, args.answer_file)
    print("over")


"""Report
# predict
python ./exercises/05-perceptron/testperceptron.py --model-file ./model/title-en-train.labeled.perceptron --input-file ./data/titles-en-test.word --answer-file ./my-answer/my-ans-title-en-test.word
predicting
over
-----------------
# test
- Just Use unigram
python ./script/grade-prediction.py data/titles-en-test.labeled ./my-answer/my-ans-title-en-test.wordAccuracy = 92.950762%

- Use unigram and bigram
python ./script/grade-prediction.py data/titles-en-test.labeled ./my-answer/my-ans-title-en-test.wordAccuracy = 93.800921%
"""