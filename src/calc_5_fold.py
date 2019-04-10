import argparse
import logging
import pdb
import pickle
import sys
import traceback
import json
import torch
from utils import MWPDataset
from calc_score import tofloat


def main(args):
    logging.basicConfig(level=logging.INFO)

    # preprocessor = Preprocessor(args.embedding_path)
    # train, valid = preprocessor.get_train_valid_dataset(args.data_path)

    with open(args.pickle_path, 'rb') as f:
        data = pickle.load(f)
        preprocessor = data['preprocessor']
        problems = data['train']._problems

    if args.arch == 'NSMv1':
        from torch_solver import TorchSolver
        solver = TorchSolver(
            preprocessor.get_word_dim(),
            args.dim_hidden,
            batch_size=args.batch_size,
            n_epochs=10000,
            device=args.device)

    if args.arch == 'NSMv2':
        from torch_solver_v2 import TorchSolverV2
        solver = TorchSolverV2(
            preprocessor.get_word_dim(),
            args.dim_hidden,
            batch_size=args.batch_size,
            n_epochs=10000,
            device=args.device)

    elif args.arch == 'seq2seq':
        from torch_seq2seq import TorchSeq2Seq
        solver = TorchSeq2Seq(
            preprocessor.get_vocabulary_size(),
            preprocessor.get_word_dim(),
            args.dim_hidden,
            embedding=preprocessor.get_embedding(),
            batch_size=args.batch_size,
            n_epochs=10000,
            device=args.device)

    accuracys = []
    for fold in range(5):
        # make valid dataset
        fold_indices = [int(len(problems) * 0.2) * i for i in range(6)]
        start = fold_indices[fold]
        end = fold_indices[fold + 1]
        valid = MWPDataset(problems[start:end],
                           preprocessor.indices_to_embeddings)

        # make prediction
        predict_filename = '{}.fold{}.{}'.format(args.output,
                                                 fold,
                                                 args.epoch)
        with open(predict_filename) as f:
            predicts = json.load(f)
            ys_ = [predict['ans'] for predict in predicts]
            ys_ = torch.tensor(ys_)

        # make answer list
        ys = torch.tensor([tofloat(p['ans']) for p in valid])
        accuracy = (ys == ys_).float().mean().item()
        print('Accuracy = {}'.format(accuracy))
        accuracys.append(accuracy)

    print('mean = {}'.format(sum(accuracys) / 5))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train the MWP solver.")
    # parser.add_argument('data_path', type=str,
    #                     help='Path to the data.')
    # parser.add_argument('embedding_path', type=str,
    #                     help='Path to the embedding.')
    parser.add_argument('pickle_path', type=str,
                        help='Path to the train valid pickle.')
    parser.add_argument('output', type=str,
                        help='Dest to dump prediction.')
    parser.add_argument('--dim_hidden', type=int, default=256,
                        help='Hidden state dimension of the encoder.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--device', default=None,
                        help='Device used to train.')
    parser.add_argument('--to_test', type=str,
                        default='valid', help='To dump train or valid.')
    parser.add_argument('--arch', type=str,
                        default='NSMv1', help='To dump train or valid.')
    parser.add_argument('--epoch', type=int,
                        default=14, help='Index of the epoch to use.')
    args = parser.parse_args()
    return args


class DumpHook:
    def __init__(self):
        self.outputs = []
        self.batch_outputs = []

    def forward_hook(self, module, inputs, outputs):
        self.batch_outputs.append(outputs)

    def flush_batch(self):
        self.outputs.append(self.batch_outputs)
        self.batch_outputs = []


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
