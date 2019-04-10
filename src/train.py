import argparse
import copy
import logging
import pdb
import pickle
import sys
import traceback
from utils import Preprocessor
from callbacks import ModelCheckpoint, MetricsLogger
from permute_stack import PermuteStackOps


def main(args):
    # preprocessor = Preprocessor(args.embedding_path)
    # train, valid = preprocessor.get_train_valid_dataset(args.data_path)

    with open(args.pickle_path, 'rb') as f:
        data = pickle.load(f)
        preprocessor = data['preprocessor']
        train, valid = data['train'], data['valid']

    if args.arch == 'NSMv1':
        from torch_solver import TorchSolver
        solver = TorchSolver(
            preprocessor.get_word_dim(),
            args.dim_hidden,
            valid=valid,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            decoder_use_state=args.decoder_use_state)

    # load model
    if args.load is not None:
        solver.load(args.load)

    if not args.five_fold:
        model_checkpoint = ModelCheckpoint(args.model_path,
                                           'loss', 1, 'all')
        metrics_logger = MetricsLogger(args.log_path)
        solver.fit_dataset(train, [model_checkpoint, metrics_logger])
    else:
        from utils import MWPDataset
        problems = train._problems
        fold_indices = [int(len(problems) * 0.2) * i for i in range(6)]
        for fold in range(5):
            train = []
            for j in range(5):
                if j != fold:
                    start = fold_indices[j]
                    end = fold_indices[j + 1]
                    train += problems[start:end]

            transform = \
                PermuteStackOps(args.revert_prob, args.transpose_prob) \
                if args.permute else None
            train = MWPDataset(train, preprocessor.indices_to_embeddings)
            logging.info('Start training fold {}'.format(fold))
            model_checkpoint = ModelCheckpoint(
                '{}.fold{}'.format(args.model_path, fold),
                'loss', 1, 'all')
            metrics_logger = MetricsLogger(
                '{}.fold{}'.format(args.log_path, fold))
            solver = TorchSolver(
                preprocessor.get_word_dim(),
                args.dim_hidden,
                valid=valid,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                learning_rate=args.learning_rate,
                device=args.device)
            solver.fit_dataset(train, [model_checkpoint, metrics_logger])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train the MWP solver.")
    # parser.add_argument('data_path', type=str,
    #                     help='Path to the data.')
    # parser.add_argument('embedding_path', type=str,
    #                     help='Path to the embedding.')
    parser.add_argument('pickle_path', type=str,
                        help='Path to the train valid pickle.')
    parser.add_argument('model_path', type=str,
                        help='Path to the model checkpoint.')
    parser.add_argument('--log_path', type=str, default='./log.json',
                        help='Path to the log file.')
    parser.add_argument('--dim_hidden', type=int, default=256,
                        help='Hidden state dimension of the encoder.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate to use.')
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Number of epochs to run.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--load', default=None, type=str,
                        help='Model to load.')
    parser.add_argument('--arch', default='NSMv1', type=str,
                        help='Model architecture.')
    parser.add_argument('--five_fold', default=False,
                        help='Wheather or not doing 5 fold cross validation',
                        action='store_true')
    parser.add_argument('--decoder_use_state', default=False,
                        help='',
                        action='store_true')
    parser.add_argument('--permute', default=False,
                        help='',
                        action='store_true')
    parser.add_argument('--revert_prob', default=0.5, type=float)
    parser.add_argument('--transpose_prob', default=0.5, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
