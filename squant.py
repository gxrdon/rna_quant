#!/usr/bin/env python

import gzip
import numpy as np
from scipy.stats import norm
from scipy import stats


# Credit to Dr. Patro for the code from line 10 to 18 to precompute effective length values
mu = 200
sd = 25
cond_means = []
for txp_len in range(1001):
        d = norm(mu, sd)
        p = np.array([d.pdf(i) for i in range(txp_len+1)])
        p /= p.sum()
        cond_mean = np.sum([i * p[i] for i in range(len(p))])
        cond_means.append(cond_mean)


def parse_file(file_input):
    transcripts = {}
    transcript_list = []
    alignments = []
    alignment_offsets = []
    precompute = []
    alignment_bool = False
    num_transcripts = 0
    count_of_aligns = 0
    d = norm(200, 25)
    D = d.cdf
    with gzip.open(file_input, 'rb') as f:
        count = 0
        for line in f:
            temp = str(line).split('\'')
            if count == 0:
                num_transcripts = int(temp[1][0: len(temp[1]) - 2])
                count = count + 1
            elif alignment_bool:
                if len(str(line).split("\\t")) == 1:
                    count_of_aligns += 1
                    alignment_offsets.append(int(temp[1][0: len(temp[1]) - 2]))
                else:
                    curr_align = temp[1].split("\\t")
                    align_name = str(curr_align[0])
                    align_ori = str(curr_align[1])
                    align_pos = int(curr_align[2])
                    align_prob = curr_align[3][0: len(curr_align[3]) - 2].split("e")
                    if len(align_prob) > 1:
                        num = float(align_prob[0])
                        power = float(align_prob[1])
                        align_prob = num * (10 ** power)
                    else:
                        align_prob = float(align_prob[0])
                    length = transcript_list[transcripts[align_name]]["length"]
                    p_2 = transcript_list[transcripts[align_name]]["p_2"]
                    p_3 = D(length - align_pos)
                    precompute.append(p_2 * p_3 * align_prob)
                    align_obj = {"index": transcripts[align_name], "ori": align_ori, "pos": align_pos, "prob": align_prob, "p_3": p_3}
                    alignments.append(align_obj)
            else:
                count = count + 1
                if count > num_transcripts:
                    alignment_bool = True
                n = temp[1].split("\\t")
                key = n[0]
                val = int(n[1][0: len(n[1]) - 2])
                transcripts[key] = count - 2
                trans_obj = {"length": val, "name": key, "p_2": 1 / get_effective_length(val)}
                transcript_list.append(trans_obj)

    return num_transcripts, transcript_list, alignments, alignment_offsets, precompute


def get_effective_length(l) -> float:
    if l >= 1000:
            return l - mu
    else:
            return l - cond_means[l]


def check_for_convergence(n_arr, n_update, num_transcripts):
    for i in range(0, num_transcripts):
        if abs(n_arr[i] - n_update[i]) > 1:
            print(str(n_arr[i]) + " = n_arr and update = " + str(n_update[i]) + " subtracted is " + str(abs(n_arr[i] - n_update[i])))
            return True
    return False


def full_model_em(align_file, output_file):
    num_transcripts, transcripts, alignments, alignment_offsets, precompute = parse_file(align_file)
    n_arr = [1/num_transcripts] * num_transcripts
    not_converged = True
    prob_totals = n_arr.copy()
    while not_converged:
        n_update = [0] * num_transcripts
        offset = 0
        for align in range(0, len(alignment_offsets)):

            # Short circuit if only one transcript instead of running EM on it
            if alignment_offsets[align] == 1:
                n_update[alignments[offset]["index"]] += 1
                offset = offset + 1
                continue
            tot = 0.0
            for idx in range(0, alignment_offsets[align]):
                curr_index = alignments[offset + idx]["index"]
                tot += n_arr[curr_index] * precompute[offset + idx]
            for idx in range(0, alignment_offsets[align]):
                curr_index = alignments[offset]["index"]
                n_update[alignments[offset]["index"]] += n_arr[curr_index] * precompute[offset] / tot
                offset += 1

        # If we find that one value hasn't converged, continue process
        ret_arr = n_update.copy()
        for i in range(0, num_transcripts):
            n_update[i] /= len(alignment_offsets)
        not_converged = check_for_convergence(prob_totals, ret_arr, num_transcripts)
        n_arr = n_update
        prob_totals = ret_arr.copy()

    write_to_file(output_file, num_transcripts, transcripts, n_arr)


def equivalence_class_em(align_file, output_file):
    num_transcripts, transcripts, alignments, alignment_offsets, precompute = parse_file(align_file)
    eq_classes = {}
    offset = 0
    for idx in range(0, len(alignment_offsets)):
        l = []
        for j in range(0, alignment_offsets[idx]):
            l.append(alignments[offset + j]["index"])
        offset += alignment_offsets[idx]
        tset = tuple(sorted(l))
        if tset in eq_classes:
            eq_classes[tset] += 1
        else:
            eq_classes[tset] = 1

    n_arr = [1 / num_transcripts] * num_transcripts
    prob_totals = n_arr.copy()
    not_converged = True
    while not_converged:
        n_update = [0] * num_transcripts
        count = 0
        for eq_class in eq_classes:

            # Short circuit if only one transcript instead of running EM on it
            count = count + 1
            if len(eq_class) == 1:
                n_update[eq_class[0]] += + 1
                continue
            summation_value = 0
            for i in range(0, len(eq_class)):
                summation_value += n_arr[eq_class[i]] * transcripts[eq_class[i]]["p_2"]
            for trans in eq_class:
                p_1 = n_arr[trans]
                p_2 = transcripts[trans]["p_2"]
                n_update[trans] += (eq_classes[eq_class] * p_1 * p_2) / summation_value

        # If we find that one value hasn't converged, continue process
        ret_arr = n_update.copy()
        for i in range(0, num_transcripts):
            n_update[i] /= len(alignment_offsets)
        not_converged = check_for_convergence(prob_totals, ret_arr, num_transcripts)
        n_arr = n_update
        prob_totals = ret_arr.copy()

    write_to_file(output_file, num_transcripts, transcripts, n_arr)


def write_to_file(output_file, num_transcripts, transcripts, n_arr):
    f = open(output_file, "w")
    for i in range(0, num_transcripts):
        f.write(str(transcripts[i]["name"]) + "\t" + str(int(round(get_effective_length(transcripts[i]["length"]), 3))) + "\t" + str(round(n_arr[i], 1)) + "\n")
    f.close()


# Credit to Dr. Patro for the following three functions that allowed me to test my estimation's accuracy
def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return np.array(actual) - np.array(predicted)


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


def get_stats(real, fake):
    """
    Function to compute the statistics for the estimation that we ran.
    """
    real_arr = []
    fake_arr = []
    count = 0
    with open(real, 'rb') as f:
        for line in f:
            if count == 0:
                count = 1
                continue
            temp = str(line).split("\\t")
            real_arr.append(float(temp[1][0: len(temp[1]) - 6]))
    with open(fake, 'rb') as f:
        for line in f:
            temp = str(line).split("\\t")
            num = temp[2].split("\\")
            fake_arr.append(float(num[0]))
    print(stats.spearmanr(real_arr, fake_arr))
    print(mse(real_arr, fake_arr))
    print(mae(real_arr, fake_arr))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EM model: Parsing and estimation of counts')
    parser.add_argument('--input', help='Input file to read from', type=str, required=True, nargs=1)
    parser.add_argument('--output', help='Output file to write results to', type=str, required=True, nargs=1)
    parser.add_argument('--eqc', help='If specified, we run the equivalence class algorithm', required=False, action='store_true')
    args = parser.parse_args()

    if args.eqc:
        equivalence_class_em(args.input[0], args.output[0])
    else:
        full_model_em(args.input[0], args.output[0])
