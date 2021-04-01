# @Time    : 2021/03/16 25:29
# @Author  : SY.M
# @FileName: clean_line.py

import re

def clean_sentences(line):
    line = re.sub('<.*?>', '', line)  # removing html tags

    # removing contractions
    line = re.sub("isn't", 'is not', line)
    line = re.sub("he's", 'he is', line)
    line = re.sub("wasn't", 'was not', line)
    line = re.sub("there's", 'there is', line)
    line = re.sub("couldn't", 'could not', line)
    line = re.sub("won't", 'will not', line)
    line = re.sub("they're", 'they are', line)
    line = re.sub("she's", 'she is', line)
    line = re.sub("There's", 'there is', line)
    line = re.sub("wouldn't", 'would not', line)
    line = re.sub("haven't", 'have not', line)
    line = re.sub("That's", 'That is', line)
    line = re.sub("you've", 'you have', line)
    line = re.sub("He's", 'He is', line)
    line = re.sub("what's", 'what is', line)
    line = re.sub("weren't", 'were not', line)
    line = re.sub("we're", 'we are', line)
    line = re.sub("hasn't", 'has not', line)
    line = re.sub("you'd", 'you would', line)
    line = re.sub("shouldn't", 'should not', line)
    line = re.sub("let's", 'let us', line)
    line = re.sub("they've", 'they have', line)
    line = re.sub("You'll", 'You will', line)
    line = re.sub("i'm", 'i am', line)
    line = re.sub("we've", 'we have', line)
    line = re.sub("it's", 'it is', line)
    line = re.sub("don't", 'do not', line)
    line = re.sub("that´s", 'that is', line)
    line = re.sub("I´m", 'I am', line)
    line = re.sub("it’s", 'it is', line)
    line = re.sub("she´s", 'she is', line)
    line = re.sub("he’s'", 'he is', line)
    line = re.sub('I’m', 'I am', line)
    line = re.sub('I’d', 'I did', line)
    line = re.sub("he’s'", 'he is', line)
    line = re.sub('there’s', 'there is', line)

    # special characters and emojis
    line = re.sub('\x91The', 'The', line)
    line = re.sub('\x97', '', line)
    line = re.sub('\x84The', 'The', line)
    line = re.sub('\uf0b7', '', line)
    line = re.sub('¡¨', '', line)
    line = re.sub('\x95', '', line)
    line = re.sub('\x8ei\x9eek', '', line)
    line = re.sub('\xad', '', line)
    line = re.sub('\x84bubble', 'bubble', line)

    # remove concated words
    line = re.sub('trivialBoring', 'trivial Boring', line)
    line = re.sub('Justforkix', 'Just for kix', line)
    line = re.sub('Nightbeast', 'Night beast', line)
    line = re.sub('DEATHTRAP', 'Death Trap', line)
    line = re.sub('CitizenX', 'Citizen X', line)
    line = re.sub('10Rated', '10 Rated', line)
    line = re.sub('_The', '_ The', line)
    line = re.sub('1Sound', '1 Sound', line)
    line = re.sub('blahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblah', 'blah blah', line)
    line = re.sub('ResidentHazard', 'Resident Hazard', line)
    line = re.sub('iameracing', 'i am racing', line)
    line = re.sub('BLACKSNAKE', 'Black Snake', line)
    line = re.sub('DEATHSTALKER', 'Death Stalker', line)
    line = re.sub('_is_', 'is', line)
    line = re.sub('10Fans', '10 Fans', line)
    line = re.sub('Yellowcoat', 'Yellow coat', line)
    line = re.sub('Spiderbabe', 'Spider babe', line)
    line = re.sub('Frightworld', 'Fright world', line)

    # removing punctuations

    punctuations = '@#!~?+&*[]-%._-:/£();$=><|{}^' + '''"“´”'`'''
    for p in punctuations:
        line = line.replace(p, f' {p} ')

    line = re.sub(',', ' , ', line)

    # ... and ..
    line = line.replace('...', ' ... ')

    if '...' not in line:
        line = line.replace('..', ' ... ')

    return line


def clean_sentences_2(line):
    line = re.sub('<.*?>', '', line)  # removing html tags

    line = re.sub('film', '', line)  # removing html tags
    line = re.sub('movie', '', line)  # removing html tags
    line = re.sub('movies', '', line)  # removing html tags
    line = re.sub('It', '', line)  # removing html tags
    line = re.sub('films', '', line)  # removing html tags
    # line = re.sub('A', '', line)  # removing html tags
    line = re.sub('In', '', line)  # removing html tags
    # line = re.sub('ll', '', line)  # removing html tags
    # line = re.sub('s', '', line)  # removing html tags
    # line = re.sub('I', '', line)  # removing html tags
    # line = re.sub('thi', '', line)  # removing html tags
    line = re.sub('The', '', line)  # removing html tags

    # removing contractions
    line = re.sub("isn't", 'is not', line)
    line = re.sub("he's", 'he is', line)
    line = re.sub("wasn't", 'was not', line)
    line = re.sub("there's", 'there is', line)
    line = re.sub("couldn't", 'could not', line)
    line = re.sub("won't", 'will not', line)
    line = re.sub("they're", 'they are', line)
    line = re.sub("she's", 'she is', line)
    line = re.sub("There's", 'there is', line)
    line = re.sub("wouldn't", 'would not', line)
    line = re.sub("haven't", 'have not', line)
    line = re.sub("That's", 'That is', line)
    line = re.sub("you've", 'you have', line)
    line = re.sub("He's", 'He is', line)
    line = re.sub("what's", 'what is', line)
    line = re.sub("weren't", 'were not', line)
    line = re.sub("we're", 'we are', line)
    line = re.sub("hasn't", 'has not', line)
    line = re.sub("you'd", 'you would', line)
    line = re.sub("shouldn't", 'should not', line)
    line = re.sub("let's", 'let us', line)
    line = re.sub("they've", 'they have', line)
    line = re.sub("You'll", 'You will', line)
    line = re.sub("i'm", 'i am', line)
    line = re.sub("we've", 'we have', line)
    line = re.sub("it's", 'it is', line)
    line = re.sub("don't", 'do not', line)
    line = re.sub("that´s", 'that is', line)
    line = re.sub("I´m", 'I am', line)
    line = re.sub("it’s", 'it is', line)
    line = re.sub("she´s", 'she is', line)
    line = re.sub("he’s'", 'he is', line)
    line = re.sub('I’m', 'I am', line)
    line = re.sub('I’d', 'I did', line)
    line = re.sub("he’s'", 'he is', line)
    line = re.sub('there’s', 'there is', line)

    # special characters and emojis
    line = re.sub('\x91The', 'The', line)
    line = re.sub('\x97', '', line)
    line = re.sub('\x84The', 'The', line)
    line = re.sub('\uf0b7', '', line)
    line = re.sub('¡¨', '', line)
    line = re.sub('\x95', '', line)
    line = re.sub('\x8ei\x9eek', '', line)
    line = re.sub('\xad', '', line)
    line = re.sub('\x84bubble', 'bubble', line)

    # remove concated words
    line = re.sub('trivialBoring', 'trivial Boring', line)
    line = re.sub('Justforkix', 'Just for kix', line)
    line = re.sub('Nightbeast', 'Night beast', line)
    line = re.sub('DEATHTRAP', 'Death Trap', line)
    line = re.sub('CitizenX', 'Citizen X', line)
    line = re.sub('10Rated', '10 Rated', line)
    line = re.sub('_The', '_ The', line)
    line = re.sub('1Sound', '1 Sound', line)
    line = re.sub('blahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblah', 'blah blah', line)
    line = re.sub('ResidentHazard', 'Resident Hazard', line)
    line = re.sub('iameracing', 'i am racing', line)
    line = re.sub('BLACKSNAKE', 'Black Snake', line)
    line = re.sub('DEATHSTALKER', 'Death Stalker', line)
    line = re.sub('_is_', 'is', line)
    line = re.sub('10Fans', '10 Fans', line)
    line = re.sub('Yellowcoat', 'Yellow coat', line)
    line = re.sub('Spiderbabe', 'Spider babe', line)
    line = re.sub('Frightworld', 'Fright world', line)

    # removing punctuations

    punctuations = '@#!~?+&*[]-%._-:/£();$=><|{}^' + '''"“´”'`'''
    for p in punctuations:
        line = line.replace(p, '')

    line = re.sub(',', '', line)

    # ... and ..
    line = line.replace('...', '')

    if '...' not in line:
        line = line.replace('..', '')

    return line