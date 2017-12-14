# -*- coding: utf-8 -*-
# @Author: Henri
# @Date:   2017-08-30 11:28:10
# @Last Modified by:   Henri
# @Last Modified time: 2017-08-30 11:58:10
import bob
import bob.measure
import logging
import argparse
import itertools
from collections import defaultdict


def labelfile(filen):
    utttolabel = {}
    with open(filen, 'r') as wp:
        for line in wp:
            utt, label = line.split()[:2]
            utttolabel[utt] = label
    return utttolabel


def scorefile(filen):
    utttoscore = {}
    with open(filen, 'r') as wp:
        for line in wp:
            utt, score = line.split()[:2]
            utt = utt.split("/")[-1]
            utttoscore[utt] = float(score)
    return utttoscore


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("evalscores", type=scorefile)
    parser.add_argument("evallabels", type=labelfile)
    parser.add_argument("-g", "--genuinelabel", default="genuine", type=str)
    parser.add_argument('-l', '--level', default=logging.WARN,
                        type=int, help="Logging level. Lower results in larger output")
    return parser.parse_args()


def labeltoscore(labels, scores):
    labscore = defaultdict(list)
    logwarns = 0
    for utt, label in labels.items():
        if not utt in scores:
            logwarns = logwarns + 1
            logging.warn("Utterance %s not found in scores" % (utt))
            continue
        score = scores[utt]
        labscore[label].append(score)
    if logwarns > 0:
        logging.warn("Encountered %i errors" % (logwarns))
    return labscore


def main():
    args = parseargs()
    logging.basicConfig(
        level=args.level, format="%(asctime)s | %(levelname)s | %(message)s")
    evallabtoscore = labeltoscore(args.evallabels, args.evalscores)
    positives = evallabtoscore[args.genuinelabel]
    for label in evallabtoscore.keys():
        # Skip genuine data ( scoring with itself)
        if label == args.genuinelabel:
            continue
        negatives = evallabtoscore[label]
        eer_label = bob.measure.eer_rocch(negatives, positives)
        logging.info("[Eval] LAB: {} EER: {:=4.3f}  ".format(
                     label, 100*eer_label))

    evalallnegatives = list(itertools.chain.from_iterable(
        [v for k, v in evallabtoscore.items() if k != args.genuinelabel]))
    evalallpositives = evallabtoscore[args.genuinelabel]
    evalthres= bob.measure.eer_threshold(evalallnegatives, evalallpositives)
    evalfar, evalfrr = bob.measure.farfrr(
        evalallnegatives, evalallpositives, evalthres)
    eval_eer = bob.measure.eer_rocch(evalallnegatives, evalallpositives)

    print("Evaluation set : FAR = {:.3f}%, FRR = {:.3f}%, EER = {:.3f}% \t ".format(
        100 * evalfar, 100 * evalfrr, eval_eer*100))


if __name__ == '__main__':
    main()
