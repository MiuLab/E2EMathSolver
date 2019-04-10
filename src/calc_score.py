import argparse
import pdb
import sys
import traceback
import json
import pickle
import re
import numpy as np


def tofloat(text):
    if '%' in text:
        return float(text[:-1]) / 100

    if '/' in text:
        match = re.search(r'(\d*)\(\((\d+)\)/\((\d+)\)\)', text)
        a = 0 if match.group(1) == '' else int(match.group(1))
        return a + int(match.group(2)) / int(match.group(3))

    return float(text)


def main(args):
    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)[args.to_test]
    with open(args.predict) as f:
        raw = json.load(f)

    predict = np.array([p['ans'] if p['ans'] is not None else 0
                        for p in raw])
    confidence = np.array([p['confidence'] for p in raw])
    answer = np.array([tofloat(d['ans']) for d in data])

    if args.retrieval is not None:
        with open(args.retrieval) as f:
            raw = json.load(f)
        retrieval = np.array([p['ans'] if p['ans'] is not None else 0
                             for p in raw])
        predict[confidence < args.threshold] = \
            retrieval[confidence < args.threshold]

    correct = np.abs(predict - answer) < args.epsilon
    accuracy = np.mean(correct)
    print('Accuracy = {}'.format(accuracy))

    correct_confidence = confidence[np.where(correct)]
    incorrect_confidence = confidence[np.where(~correct)]
    print('Correct Confidence mean={}, std={}'
          .format(np.mean(correct_confidence),
                  np.std(correct_confidence))
          )
    print('Incorrect Confidence mean={}, std={}'
          .format(np.mean(incorrect_confidence),
                  np.std(incorrect_confidence))
          )

    if args.log is not None:
        with open(args.log, 'w') as f:
            log = {
                'accuracy': accuracy,
                'correct': list(predict - answer)
            }
            json.dump(log, f, indent='    ')


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy")
    parser.add_argument('pickle', type=str,
                        help='')
    parser.add_argument('predict', type=str,
                        help='')
    parser.add_argument('--to_test', type=str,
                        default='valid', help='To dump train or valid.')
    parser.add_argument('--epsilon', type=float,
                        default=1e-4,
                        help='Error that is tolerant as correct.')
    parser.add_argument('--log', type=str, default=None,
                        help='Destination of log file.')
    parser.add_argument('--retrieval', type=str, default=None,
                        help='')
    parser.add_argument('--threshold', type=float, default=-0.6)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
