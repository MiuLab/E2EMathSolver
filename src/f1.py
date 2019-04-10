import argparse
import pdb
import sys
import traceback
import pickle
import json
import pickle
import os
from define import OPERATIONS
from calc_5_fold import tofloat


def main(args):
    with open(args.data_path, 'rb') as f:
        train = pickle.load(f)['train']._problems

    ops = ['noop', 'gv', '+', '-', '*', '/', '=']
    for sample in train:
        mapping = ops + [str(c) for c in sample['constants']] + ['x0']
        sample['equations'] = [mapping[s] for s in sample['operations']]
        sample['equations'] = list(filter(lambda x: x not in ['noop', 'gv'],
                                          sample['equations']))

    predicts = []
    for i in range(5):
        predict_path = os.path.join(args.model_path,
                                    'predict.json.fold{}.{}'
                                    .format(i, args.epoch))
        with open(predict_path) as f:
            predicts += json.load(f)

    correct_indices, incorrect_indices = [], []
    for i, (p, a) in enumerate(zip(predicts, train)):
        if p['ans'] == tofloat(a['ans']):
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)

    ps, rs, f1s = [], [], []
    for predict, answer in zip(predicts, train):
        eqp = eval(predict['equations'])
        eqa = answer['equations']
        # eqp = collect_subtrees(eqp)
        # eqa = collect_subtrees(eqa)
        # eqp = list(filter(lambda x: x not in ['+', '-', '*', '/', '='], eqp))
        # eqa = list(filter(lambda x: x not in ['+', '-', '*', '/', '='], eqa))
        eqp, eqa = set(eqp), set(eqa)
        precision = len(eqp & eqa) / len(eqp)
        recall = len(eqp & eqa) / len(eqa)
        f1 = precision * recall / (precision + recall) * 2
        ps.append(precision)
        rs.append(recall)
        f1s.append(f1)

    print('accuracy = {}'.format(len(correct_indices) / len(predicts)))
    print('All, {}, {}, {}'
          .format(sum(ps) / len(ps),
                  sum(rs) / len(rs),
                  sum(f1s) / len(f1s)
                  )
          )

    cps, crs, cf1s = ([ps[i] for i in correct_indices],
                      [rs[i] for i in correct_indices],
                      [f1s[i] for i in correct_indices])
    print('Correct, {}, {}, {}'
          .format(sum(cps) / len(cps),
                  sum(crs) / len(crs),
                  sum(cf1s) / len(cf1s)
                  )
          )
    ips, irs, if1s = ([ps[i] for i in incorrect_indices],
                      [rs[i] for i in incorrect_indices],
                      [f1s[i] for i in incorrect_indices])
    print('Incorrect, {}, {}, {}'
          .format(sum(ips) / len(ips),
                  sum(irs) / len(irs),
                  sum(if1s) / len(if1s)
                  )
          )


def collect_subtrees(ops):
    subtrees = []
    stack = []
    for op in ops:
        if op in ['+', '-', '*', '/', '=']:
            opd1, opd2 = stack.pop(), stack.pop()

            if op in ['+', '*'] and opd2 > opd1:
                expr = opd1 + op + opd2

            expr = '({} {} {})'.format(opd2, op, opd1)
            
            stack.append(expr)
            subtrees.append(expr)
        else:
            stack.append(op)
            subtrees.append(op)

    return subtrees


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('data_path', type=str,
                        help='')
    parser.add_argument('model_path', type=str,
                        help='')
    parser.add_argument('epoch', type=int)
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
