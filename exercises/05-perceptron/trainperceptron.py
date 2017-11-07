import argparse
import pickle

from collections import defaultdict


def create_features(x):
    phi = defaultdict(int)
    words = x.split()
    last_word = "<s>"
    for w in words:
        phi["UNI:" + w] += 1  # We add "UNI:" to indicate unigrams
        phi["BI:"+last_word+w] += 1     # "BI:" to indicate unigrams
        last_word = w
    # print(phi)
    return phi


def predict_one(weight, phi):
    score = 0
    for key in phi:
        if key in weight:
            score += weight[key] * phi[key]
    #    else: score += 0 * phi[key]
    if score >= 0:
        return 1    # positive
    else:
        return -1   # negative


def update_weight(weight, phi, y):
    for key in phi:
        weight[key] += phi[key] * y


def train(training_file, model_file, iter_num):
    weight = defaultdict(int)

    data = None
    with open(training_file, 'r') as f:
        data = f.readlines()
    print("data length:", len(data))
    print("training...")
    for i in range(iter_num):
        total_count, right_count = 0, 0
        for line in data:
            y_x = line.split("\t")
            y = int(y_x[0])
            x = y_x[1]
            phi = create_features(x)
            predicted_y = predict_one(weight, phi)
            if predicted_y != y:
                update_weight(weight, phi, y)
            else:
                right_count += 1
            total_count += 1
        print("Acc: ", right_count / total_count)

    print("saving model")

    with open(model_file, 'wb') as f:
        pickle.dump(weight, f)

    print("saved")


if __name__ == "__main__":
    # python  ./exercises/05-perceptron/trainperceptron.py --training-file./data/titles-en-train.labeled \
    # --model-file ./model/title-en-train.labeled.perceptron --iter-num 5
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-file", type=str)
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--iter-num", type=int)
    args = parser.parse_args()

    train(args.training_file, args.model_file, args.iter_num)


"""Report
- Just use unigram
python ./exercises/05-perceptron/trainperceptron.py --training-file ./data/titles-en-train.labeled --model-file ./model/title-en-train.labeled.perceptron --iter-num 30
data length: 11288
training...
Acc:  0.8784549964564139
Acc:  0.9270021261516654
Acc:  0.9410878809355068
Acc:  0.954376328844791
Acc:  0.96243798724309
Acc:  0.9690822111977321
Acc:  0.9728915662650602
Acc:  0.9789156626506024
Acc:  0.9809532246633593
Acc:  0.9832565556343019
Acc:  0.9874202693125443
Acc:  0.9859142452161588
Acc:  0.986268603827073
Acc:  0.9895464209780298
Acc:  0.9911410347271439
Acc:  0.9904323175053154
Acc:  0.9932671863926293
Acc:  0.9914068036853295
Acc:  0.9930900070871722
Acc:  0.9920269312544295
Acc:  0.9938873139617292
Acc:  0.9950389794472005
Acc:  0.9951275690999292
Acc:  0.9961020552799433
Acc:  0.9953047484053863
Acc:  0.9970765414599575
Acc:  0.9948618001417434
Acc:  0.9944188518781006
Acc:  0.9962792345854005
Acc:  0.9980510276399717
saving model
saved
----------------------------
- Use unigram and bigram
python ./exercises/05-perceptron/trainperceptron.py --training-file ./data/titles-en-train.labeled --model-file ./model/title-en-train.labeled.perceptron --iter-num 30data length: 11288
training...
Acc:  0.8918320340184267
Acc:  0.9507441530829199
Acc:  0.9707654145995748
Acc:  0.9798901488306165
Acc:  0.9870659107016301
Acc:  0.9906980864635011
Acc:  0.9934443656980865
Acc:  0.9950389794472005
Acc:  0.996367824238129
Acc:  0.9952161587526577
Acc:  0.9958362863217576
Acc:  0.9987597448618002
Acc:  0.9972537207654146
Acc:  0.9982282069454288
Acc:  0.9969879518072289
Acc:  0.9991141034727143
Acc:  0.9998228206945429
Acc:  0.9990255138199858
Acc:  0.9994684620836286
Acc:  0.999202693125443
Acc:  0.9984053862508859
Acc:  0.9988483345145287
Acc:  0.9990255138199858
Acc:  0.9993798724309001
Acc:  0.9995570517363572
Acc:  0.9998228206945429
Acc:  0.9996456413890857
Acc:  1.0
Acc:  1.0
Acc:  1.0
saving model
saved
"""

