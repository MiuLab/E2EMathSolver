import argparse
import logging
import pdb
import sys
import traceback
import pickle
import json


def main(args):

    if args.dataset == 'Dolphin18k':
        from utils import Preprocessor as Preprocessor
    elif args.dataset == 'Math23k':
        from utils import Math23kPreprocessor as Preprocessor
    else:
        logging.error('Not compitable dataset!')
        return

    if args.index is not None:
        with open(args.index) as f:
            shuffled_index = json.load(f)
    else:
        shuffled_index = None

    preprocessor = Preprocessor(args.embedding_path)

    train, valid = preprocessor.get_train_valid_dataset(args.data_path,
                                                        args.valid_ratio,
                                                        index=shuffled_index,
                                                        char_based=args.char_based)

    with open(args.output, 'wb') as f:
        pickle.dump({'train': train,
                     'valid': valid,
                     'preprocessor': preprocessor}, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('data_path', type=str,
                        help='Path to the data.')
    parser.add_argument('embedding_path', type=str,
                        help='Path to the embedding.')
    parser.add_argument('output', type=str,
                        help='Path to the output pickle file.')
    parser.add_argument('--dataset', type=str, default='Math23k',
                        help='[Math23k|Dolphin18k]')
    parser.add_argument('--valid_ratio', type=float, default=0.2,
                        help='Ratio of data used as validation set.')
    parser.add_argument('--index', type=str, default=None,
                        help='JSON file that stores shuffled index.')
    parser.add_argument('--char_based', default=False, action='store_true',
                        help='If segment the text based on char.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
