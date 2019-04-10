import argparse
import pdb
import re
import sys
import traceback
from enum import Enum


num_map = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '兩': 2
}
order_map1 = {
    '十': 10, '百': 100, '千': 1000
}
order_map2 = {
    '萬': 10000, '億': 100000000, '兆': 1000000000000
}
symbol_pattern = re.compile(r'^[ \d\+\-\*/\.\(\)]*$')
word_pattern = re.compile(
    r'^( 等於|多少|次方|加上|減掉|減去|乘以|乘上|除以|然後|會是|是多少|'
    r'[後再點的一二三四五六七八九十百千萬億兆加減乘除'
    r'\d\+\-\*/\.^\(\)=])*[\?？]?$')
num_pattern = re.compile('[零一二三四五六七八九十百千萬億兆點兩]+')


class ProblemType(Enum):
    SYMBOL = 'pure symbol'
    WORD = 'math word problem'
    SPOKEN = 'spoken equation'


def chinese_to_number(inputs):
    if '點' in inputs:
        round_part, float_part = inputs.split('點')
    else:
        round_part, float_part = inputs, ''

    num_ord1 = 0
    num_ord2 = 0
    number = 0

    # parse the round part
    if round_part[0] == '十':
        num_ord2 = 10
    else:
        num_ord1 = num_map[round_part[0]]

    for i, char in enumerate(round_part[1:]):
        if char in num_map:
            # deal with cases "一萬五、一百五"
            if i == len(round_part[1:]) - 1:  # last character
                if round_part[i] in order_map1:
                    num_ord1 = num_map[char] * order_map1[round_part[i]] // 10
                elif round_part[i] in order_map2:
                    num_ord1 = num_map[char] * order_map2[round_part[i]] // 10
                else:
                    num_ord1 = num_ord1 * 10 + num_map[char]
            else:
                num_ord1 = num_ord1 * 10 + num_map[char]
        elif char in order_map1:
            num_ord2 += num_ord1 * order_map1[char]
            num_ord1 = 0
        elif char in order_map2:
            number += (num_ord2 + num_ord1) * order_map2[char]
            num_ord1 = 0
            num_ord2 = 0
    number += num_ord2 + num_ord1

    # parse the float part
    base = 0.1
    for char in float_part:
        number += num_map[char] * base
        base *= 0.1

    return number


def spoken_to_symbol(spoken):
    spoken = re.sub(r'的([點零一二三四五六七八九十百千萬億兆]*)次方', r'**\1', spoken)
    spoken = re.sub(r'[上去以掉再然後會是多少等於=？]', '', spoken)
    operator_map = {
        '加': '+',
        '減': '-',
        '乘': '*',
        '除': '/',
    }
    for k, v in operator_map.items():
        spoken = spoken.replace(k, v)

    operands = re.split(r'\*\*|[\+\-\*/]', spoken)
    operators = re.findall(r'\*\*|[\+\-\*/]', spoken)
    equation = str(chinese_to_number(operands[0]))
    for operand, operator in zip(operands[1:], operators):
        equation += operator + str(chinese_to_number(operand))

    return equation


def classify_input(inputs):
    """ Classify if the input is math word problem or spoken equation or pure
        symbol.
    """
    if re.match(symbol_pattern, inputs):
        return ProblemType.SYMBOL

    if re.match(word_pattern, inputs):
        return ProblemType.SPOKEN

    return ProblemType.WORD


def main(args):
    inputs_type = classify_input(args.inputs)
    print('input type = {}, equation = {}, answer = {}'
          .format(inputs_type.name, '', ''))


def _parse_args():
    parser = argparse.ArgumentParser(description="input")
    parser.add_argument('inputs', type=str,
                        help='')
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
