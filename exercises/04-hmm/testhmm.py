import argparse
import math


def load_model(model_file):
    transition = dict()
    emission = dict()
    possible_tags = dict()

    with open(model_file, 'r') as f:
        for line in f:
            kind, context, word, prob = tuple(line.split())
            prob = float(prob)
            possible_tags[context] = 1
            if kind == "T":
                transition[context + " " + word] = prob
            else:
                emission[context + " " + word] = prob

    return transition, emission, possible_tags


def forward(words, model_file):
    # Forward step
    transition, emission, possible_tags = load_model(model_file)
    lambda_1 = 0.95
    vocab_size = 1e10
    best_scores = dict()
    best_edges = dict()
    length = len(words)
    best_scores["0 <s>"] = 0    # Start with <s>
    best_edges["0 <s>"] = None
    tags = possible_tags.keys()
    for i in range(length):
        for prev in tags:
            for next in tags:
                prev_key = str(i) + " " + prev
                trans_key = prev + " " + next
                emit_key = next + " " + words[i]
                if prev_key in best_scores and trans_key in transition:
                    prob_emit = (1 - lambda_1) * (1 / vocab_size)
                    if emit_key in emission:
                        prob_emit += lambda_1 * emission[emit_key]
                    score = best_scores[prev_key] + (-math.log2(transition[trans_key])) + \
                        -math.log2(prob_emit)
                    cur_key = str(i + 1) + " " + next
                    if cur_key not in best_scores or score < best_scores[cur_key]:
                        best_scores[cur_key] = score
                        # edge: prev_key——cur_key
                        best_edges[cur_key] = prev_key
    # do the same for </s>
    for prev in tags:
        next = "</s>"   # the end tag
        prev_key = str(length) + " " + prev
        trans_key = prev + " " + next
        emit_key = next + " " + ""  # no word emited by tag </s>
        if prev_key in best_scores and trans_key in transition:
            prob_emit = (1 - lambda_1) * (1 / vocab_size)
            if emit_key in emission:
                prob_emit += lambda_1 * emission[emit_key]
            score = best_scores[prev_key] + (-math.log2(transition[trans_key])) + \
                -math.log2(prob_emit)
            cur_key = str(length + 1) + " " + next
            if cur_key not in best_scores or score < best_scores[cur_key]:
                best_scores[cur_key] = score
                best_edges[cur_key] = prev_key  # edge: prev_key——cur_key

    return best_edges


def backward(length, best_edges):
    tags = []
    last_key = str(length + 1) + " </s>"
    begin_key = "0 <s>"
    next_edge = best_edges[last_key]

    while next_edge != begin_key:
        tags.append(next_edge.split()[1])
        next_edge = best_edges[next_edge]

    tags.reverse()
    return tags


def viterbi(words, model_file):
    best_edges = forward(words, model_file)
    tags = backward(len(words), best_edges)
    return ' '.join(tags)


def test(model_file, input_file, answer_file):
    ans = []
    with open(input_file, 'r') as f:
        for line in f:
            words = line.split()
            pos_ans = viterbi(words, model_file)
            ans.append(pos_ans + "\n")
    with open(answer_file, 'w') as f:
        f.writelines(ans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--answer-file", type=str)
    args = parser.parse_args()

    test(args.model_file, args.input_file, args.answer_file)
