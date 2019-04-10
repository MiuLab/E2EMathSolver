import json
import logging
import random
import re
import torch
from torch.utils.data import Dataset
from define import OPERATIONS
from text_num_utils import isfloat, number_pattern, text2num
import pdb


class MWPDataset(Dataset):
    """ Dataset of math word problems.

    Args:
        problems (list): A list containing objects that have
            - text (list): List of indices of the words in a math word problem.
            - constants (list): Value of the constant in text.
            - constant_indices (list): Indices of the constant in text.
            - operations (list): List of OPERATIONS to use to solve a math word
                problem.
            - text_len (int): Actural length of the text.
            - ans (list): List of solutions.
            Note that the lists `text` and `operations` should be
            padded.
        encode_fn (function): A function that converts list of word indices to
            tensor consisting of word embeddings.
    """
    def __init__(self, problems, encode_fn):
        self._encode_fn = encode_fn
        self._problems = problems

    def __len__(self):
        return len(self._problems)

    def __getitem__(self, index):
        # copy problem from list
        problem = dict(self._problems[index])
        problem['indice'] = problem['text']
        problem['text'] = self._encode_fn(problem['text'])
        return problem


class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embedding_path, max_text_len=150, max_op_len=50):
        logging.info('loading embedding...')
        self.num_token = '<num>'
        self._word_dict, self._embeddings = load_embeddings(embedding_path)
        self._gen_num_embedding()
        self._max_text_len = max_text_len
        self._max_op_len = max_op_len

    def get_train_valid_dataset(self, data_path, valid_ratio=0.2,
                                index=None, char_based=False):
        """ Load data and return MWPDataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
            valid_ratio (float): Ratio of the data to used as valid data.
        """
        logging.info('loading dataset...')
        with open(data_path) as fp_data:
            raw_problems = json.load(fp_data)

        logging.info('preprocessing data...')
        processed = []
        n_fail = 0
        for problem in raw_problems:
            try:
                processed_problem = self.preprocess_problem(problem,
                                                            char_based=char_based)
                if (processed_problem['op_len'] > 0 and
                        processed_problem['op_len'] < 25):
                    processed.append(processed_problem)
            except (ValueError,
                    ZeroDivisionError,
                    EquationParsingException) as err:
                n_fail += 1
                text_key = 'segmented_text' \
                    if 'segmented_text' in problem else 'text'
                equation_key = 'equation' \
                    if 'equation' in problem else 'equations'
                logging.warn('Fail to parse:\n'
                             'error: {}\n'
                             'text: {}\n'
                             'equation: {}\n'.format(
                                 err,
                                 problem[text_key],
                                 problem[equation_key]))

        if n_fail > 0:
            logging.warn('Fail to parse {} problems!'.format(n_fail))

        logging.info('Parsed {} problems.'.format(len(processed) - n_fail))

        if index is None:            
            random.shuffle(processed)
        else:
            processed = [processed[i] for i in index]
        n_valid = int(len(processed) * valid_ratio)

        return (MWPDataset(processed[n_valid:], self.indices_to_embeddings),
                MWPDataset(processed[:n_valid], self.indices_to_embeddings))

    def preprocess_problem(self, problem, pad=True, char_based=False):
        """ Preprocess problem to convert a problem to the form required by solver.

        Args:
            problem (dict): A dictionary containing (loaded from Dolphin18k)
                - text (str): Text part in a math word problem.
                - ans (str): Optional, required only when training.
                - equations (str): Optional, required only when training.
                - id (str): Problem ID.

        Return:
            dict: A dictionary containing
                - text (list): List of indices of the words in a math word
                    problem. It is padded to `max_text_len`.
                - text_len (int): Actural length of the text.
                - constants (list): Value of the constant in text.
                - constant_indices (list): Indices of the constant in text.
                - operations (tensor): OPERATIONS to use to solve a math word
                    problem. It is padded to `max_op_len`.
                - op_len (int): Actural number of operations.
                - ans (list): List of solutions.
                - id (str): Problem ID.
        """
        processed = {}
        processed['ans'] = problem['ans']

        # replace numbers in words with digits
        text = replace_number_with_digits(problem['text'].lower())
        tokens = sentence_to_tokens(text, char_based)

        processed['id'] = problem['id']

        # extract number tokens
        processed['constants'] = [1, 3.14]
        processed['constant_indices'] = []
        for i, token in enumerate(tokens):
            if isfloat(token):
                processed['constants'].append(float(token))
                processed['constant_indices'].append(
                    min(i, self._max_text_len - 1))
                tokens[i] = self.num_token

        # get actural length before padding
        processed['text_len'] = min(self._max_text_len, len(tokens))

        # pad with '</s>'
        processed['text'] = self.tokens_to_indices(tokens)
        processed['text'] = pad_to_len(processed['text'],
                                       self._word_dict['</s>'],
                                       self._max_text_len)

        # construct ground truth stack operations if 'euqations' is provided
        if 'equations' in problem:
            processed['operations'] = \
                self.construct_stack_ops(problem['equations'],
                                         processed['constants'],
                                         processed['constant_indices'])
            processed['op_len'] = min(self._max_op_len,
                                      len(processed['operations']))
            processed['operations'] = pad_to_len(
                processed['operations'], OPERATIONS.NOOP, self._max_op_len)
            processed['operations'] = \
                torch.Tensor(processed['operations']).long()

        return processed

    def construct_stack_ops(self, unk_equations, constants, constant_indices):
        """ Construct stack operations that build the given equations.

        Args:
            unk_equations (str): `equations` attribute in Dolphin18k dataset.
            constants (list): Values of the constants in the text.
            constant_indices (list): Location (indices) of the constant in the
                text.

        Return:
            - operations (list): List of OPERATIONS to use to solve a math word
                problem.
        """
        # split equations string
        _, *equations = unk_equations.split('\r\nequ: ')

        # find all unknown variables that appear in equations
        # (`unkn` part of `euqations` attribute in the dataset may not
        #  contain all unknow vars in the `equ` part)
        unknowns = []
        for match in re.finditer('[a-z]', ' '.join(equations)):
            if match.group() not in unknowns:
                unknowns.append(match.group())

        # accumulator
        operations = []

        # generate variable based on number of unknowns
        for _ in range(len(unknowns)):
            operations.append(OPERATIONS.GEN_VAR)

        # prepare list of operands
        operands = constants + unknowns

        # mapping from operator token to its encoding
        op_map = {
            '+': OPERATIONS.ADD,
            '-': OPERATIONS.SUB,
            '*': OPERATIONS.MUL,
            '/': OPERATIONS.DIV,
            '=': OPERATIONS.EQL
        }

        # start parsing equations
        for equation in equations:

            # substitute fraction in equation with float
            for match in re.finditer(r'\(([0-9]+)/([0-9]+)\)', equation):
                frac = int(match.group(1)) / int(match.group(2))
                if frac in operands:
                    equation = equation.replace(match.group(), str(frac))

            postfix = infix2postfix(equation)
            for token in postfix:
                # deal with operators
                if token in op_map:
                    operations.append(op_map[token])

                # deal with operands
                else:
                    if isfloat(token):
                        token = float(token)
                    operations.append(operands.index(token) + OPERATIONS.N_OPS)

        return operations

    def tokens_to_indices(self, tokens):
        word_indices = []
        for w in tokens + ['</s>']:
            if w in self._word_dict:
                word_indices.append(self._word_dict[w])
            else:
                word_indices.append(self._word_dict['<unk>'])

        return word_indices

    def build_rev_dict(self):
        self._rev_dict = [None] * len(self._word_dict)
        for k, v in self._word_dict.items():
            self._rev_dict[v] = k

    def indices_to_tokens(self, indices):
        return [self._rev_dict[i] for i in indices]

    def indices_to_embeddings(self, indices):
        return torch.stack([self._embeddings[i] for i in indices], dim=0)

    def get_word_dim(self):
        return self._embeddings.shape[1]

    def get_vocabulary_size(self):
        return self._embeddings.shape[0]

    def get_embedding(self):
        return self._embeddings

    def _gen_num_embedding(self):
        """ Generate embedding for number token.
        """
        self._word_dict[self.num_token] = self._embeddings.size(0)
        num_indices = [self._word_dict[num]
                       for num in '1234567890']
        num_embedding = torch.mean(self._embeddings[num_indices],
                                   dim=0, keepdim=True)
        self._embeddings = torch.cat([self._embeddings, num_embedding], dim=0)


class Math23kPreprocessor(Preprocessor):
    def __init__(self, *args, **kwargs):
        super(Math23kPreprocessor, self).__init__(*args, **kwargs)

    def preprocess_problem(self, problem, char_based=False):
        """ Preprocess problem to convert a problem to the form required by solver.

        Args:
            problem (dict): A dictionary containing (loaded from Dolphin18k)
                - id (str): Problem ID.
                - segmented_text (str): Segmented text part in a math word
                    problem.
                - ans (str): Optional, required only when training.
                - equation (str): Optional, required only when training.

        Return:
            dict: A dictionary containing
                - text (list): List of indices of the words in a math word
                    problem. It is padded to `max_text_len`.
                - text_len (int): Actural length of the text.
                - constants (list): Value of the constant in text.
                - constant_indices (list): Indices of the constant in text.
                - operations (tensor): OPERATIONS to use to solve a math word
                    problem. It is padded to `max_op_len`.
                - op_len (int): Actural number of operations.
                - ans (list): List of solutions.
        """
        intermediate = {
            'id': problem['id'],
            'text': problem['segmented_text'],
            'ans': problem['ans'],
            'equations': '\r\nequ: ' + problem['equation']
            }

        for match in re.finditer(r'(\d*\.?\d+)%', intermediate['equations']):
            intermediate['equations'] = intermediate['equations'].replace(
                match.group(),
                str(float(match.group(1)) / 100))
        intermediate['equations'] = intermediate['equations'].replace('[', '(')
        intermediate['equations'] = intermediate['equations'].replace(']', ')')
        return super(Math23kPreprocessor, self) \
            .preprocess_problem(intermediate, char_based=char_based)


def load_embeddings(embedding_path):
    word_dict = {}
    with open(embedding_path) as fp:
        embedding = []

        row1 = fp.readline()
        # if the first row is not header
        if not re.match('^[0-9]+ [0-9]+$', row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        for i, line in enumerate(fp):
            cols = line.rstrip().split(' ')
            word = cols[0]
            word_dict[word] = i
            embedding.append([float(v) for v in cols[1:]])

        if '</s>' not in word_dict:
            word_dict['</s>'] = len(embedding)
            embedding.append([0] * len(embedding[0]))

        if '<unk>' not in word_dict:
            word_dict['<unk>'] = len(embedding)
            embedding.append([0] * len(embedding[0]))

    return word_dict, torch.Tensor(embedding)


def sentence_to_tokens(sentence, char_based=False):
    """ Normalize text and tokenize to tokens.
    """
    if not char_based:
        sentence = sentence.replace('. ', ' . ')
        sentence = re.sub('.$', ' .', sentence)
        sentence = sentence.replace(', ', ' , ')
        sentence = sentence.replace('$', '$ ')
        sentence = sentence.replace('?', ' ?')
        sentence = sentence.replace('!', ' !')
        sentence = sentence.replace('"', ' "')
        sentence = sentence.replace('\'', ' \'')
        sentence = sentence.replace(';', ' ;')
        sentence = sentence.strip().lower().replace('\n', ' ')
    else:
        sentence = sentence.replace(' ', '')
        sentence = ' '.join(sentence)
        sentence = re.sub(r'(?<=[\d\.]) (?=[\d\.])', '', sentence)

    return sentence.split(' ')


def pad_to_len(arr, pad, max_len):
    """ Pad and truncate to specific length.

    Args:
        arr (list): List to pad.
        pad: Element used to pad.
        max_len: Max langth of arr.
    """
    padded = [pad] * max_len
    n_copy = min(len(arr), max_len)
    padded[:n_copy] = arr[:n_copy]
    return padded


def infix2postfix(infix):
    """ Convert infix equation to postfix representation.

    Args:
        infix (str): Math expression.
    """
    infix = re.sub(r'([\d\.]+) *([a-z]+)', r'\1 * \2', infix).strip()

    # add spaces between numbers and operators
    infix = re.sub(r'([+\*/\-\(\)=])', r' \1 ', infix).strip()

    # remove consequitive spaces in the expression
    infix = re.sub(r' +', r' ', infix)

    # so now numbers and operators are seperated by exactly one space
    tokens = infix.split(' ')

    # deal with negative symbol, so it will not be seen as minus latter
    redundant_minus_indices = []
    for i, token in enumerate(tokens):
        if token == '-' and (i == 0 or
                             (i > 0 and tokens[i-1] in '=+-*/(')):
            tokens[i + 1] = str(-float(tokens[i + 1]))
            redundant_minus_indices.append(i)
    for i in redundant_minus_indices[::-1]:
        del tokens[i]

    # convert tokens to postfix
    postfix = []
    operator_stack = []
    operators = '=()+-*/'
    try:
        for token in tokens:
            if token not in operators:
                postfix.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack[-1] != '(':
                    postfix.append(operator_stack.pop())

                operator_stack.pop()
            else:
                while (len(operator_stack) > 0 and
                       operators.index(operator_stack[-1])
                       >= operators.index(token)):
                    op = operator_stack.pop()
                    # if op not in '()':
                    postfix.append(op)
                operator_stack.append(token)
    except BaseException as exception:
        raise EquationParsingException(''.join(infix),
                                       exception)

    while len(operator_stack) > 0:
        op = operator_stack.pop()
        if op not in '()':
            postfix.append(op)

    return postfix


def replace_number_with_digits(text):
    text = text.replace('$', '$ ') \
               .replace('/', ' / ') \
               .replace('a half', '1.5')
    text = text.replace('twice', '2 times')
    text = text.replace('double', '2 times')

    for match in re.finditer('([0-9]+) ([0-9]+) */ *([0-9]+)', text):
        frac = int(match.group(1)) + int(match.group(2)) / int(match.group(3))
        text = text.replace(match.group(), str(frac))

    for match in re.finditer(r'\(([0-9]+) */ *([0-9]+)\)', text):
        frac = int(match.group(1)) / int(match.group(2))
        text = text.replace(match.group(), str(frac))

    for match in re.finditer(r'([0-9]+\.)?[0-9]+%', text):
        percent_text = match.group()
        float_text = str(float(percent_text[:-1]) / 100)
        text = text.replace(percent_text,
                            float_text)

    for match in re.finditer('\\d{1,3}(,\\d{3})+', text):
        match_text = match.group()
        text = text.replace(match_text,
                            match_text.replace(',', ''))

    for num_text in number_pattern.finditer(text):
        text = text.replace(num_text.group(),
                            str(text2num(num_text.group())))

    text = re.sub(r'(-?\d+.\d+|\d+)', r' \1 ', text)
    text = re.sub(r' +', ' ', text)
    return text


class EquationParsingException(BaseException):
    def __init__(self, equation, exception):
        self.equation = equation
        self.exception = exception

    def __repr__(self):
        return 'Fail to parse equation "{}".\nError: {}'.format(
            self.equation,
            self.exception)
