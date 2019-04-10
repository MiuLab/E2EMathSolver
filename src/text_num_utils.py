import pdb
import re


_to_9 = '(zero|one|two|three|four|five|six|seven|eight|nine)'
xty = '(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
_to_19 = '(ten|eleven|twelve|thirteen|fourteen|fifteen' \
    '|sixteen|seventeen|eighteen|nineteen|{})'.format(_to_9)
_to_99 = '(({xty}[ -]{to_9})|{xty}|{to_19})'.format(
    to_19=_to_19,
    to_9=_to_9,
    xty=xty)
_to_999 = '({to_9} hundred( (and )?{to_99})?|{to_99})'.format(
    to_9=_to_9, to_99=_to_99)
_to_999999 = '({to_999} thousand( (and)? {to_999})?|{to_999})'.format(
    to_999=_to_999)
_to_9x9 = '({to_999999} million( (and)? {to_999999})?|{to_999999})'.format(
    to_999999=_to_999999)
_to_9x12 = '({to_9x9} billion( (and)? {to_9x9})?|{to_9x9})'.format(
    to_9x9=_to_9x9)

_fraction = '({to_19}-(second|third|fourth|fifth|sixth|seventh|eighth|ninth|' \
            'tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|' \
            'sixteenth|seventeenth|eighteenth|nineteenth|twentyth)|' \
            'half|quarter)'.format(
                to_19=_to_19)

_numbers = '(({to_9x12} and )?{fraction}|{to_9x12})'.format(
    to_9x12=_to_9x12, fraction=_fraction)

fraction_pattern = re.compile(_fraction)
number_pattern = re.compile(_numbers)


def text2num(text):
    """ Convert text to number.
    """
    base = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60,
        'seventy': 70,
        'eighty': 80,
        'ninety': 90,
        'twice': 2,
        'half': 0.5,
        'quarter': 0.25}

    scale = {
        'thousand': 1000,
        'million': 1000000,
        'billion': 1000000000}

    order = {
        'second': 2,
        'thirds': 3,
        'fourths': 4,
        'fifths': 5,
        'sixths': 6,
        'sevenths': 7,
        'eighths': 8,
        'nineths': 9,
        'tenths': 10,
        'elevenths': 11,
        'twelfths': 12,
        'thirteenths': 13,
        'fourteenths': 14,
        'fifteenths': 15,
        'sixteenths': 16,
        'seventeenths': 17,
        'eighteenths': 18,
        'nineteenths': 19,
        'twentyths': 20,
        'third': 3,
        'fourth': 4,
        'fifth': 5,
        'sixth': 6,
        'seventh': 7,
        'eighth': 8,
        'nineth': 9,
        'tenth': 10,
        'eleventh': 11,
        'twelfth': 12,
        'thirteenth': 13,
        'fourteenth': 14,
        'fifteenth': 15,
        'sixteenth': 16,
        'seventeenth': 17,
        'eighteenth': 18,
        'nineteenth': 19,
        'twentyth': 20}

    tokens = []
    for token in text.split(' '):
        if token == 'and':
            continue
        elif '-' in token:
            if token.split('-')[-1] in order:
                tokens.append(token)
            else:
                tokens += token.split('-')
        else:
            tokens.append(token)

    result = 0
    leading = 0
    for token in tokens:
        if token in base:
            leading += base[token]
        elif token == 'hundred':
            leading *= 100
        elif token in scale:
            result += leading * scale[token]
            leading = 0
        elif token in order:
            result += leading / order[token]
            leading = 0
        elif '-' in token:
            numerator, denominator = token.split('-')
            result += base[numerator] / order[denominator]
    result += leading

    return result


def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
