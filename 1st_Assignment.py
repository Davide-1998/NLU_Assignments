import spacy


def rootToTokenPath(sentence):
    # Finds and prints dependency path from Root to specified token

    nlp = spacy.load('en_core_web_sm')  # Create language loading a pipline
    # Create doc object by feeding the input sentence to the pipeline
    doc = nlp(sentence)

    listOfDependenciesPath = []

    for token in doc:
        listOfDependencies = []
        while token.dep_ != 'ROOT':
            listOfDependencies.insert(0, token)
            token = token.head
        listOfDependencies.insert(0, token)

        for el in listOfDependencies:
            print(' -[{}]-> {}'.format(el.dep_, el.text), end='')
        print('\n')

        listOfDependenciesPath.append(listOfDependencies)
    return listOfDependenciesPath


def subtreeOfDependents(sentence, output=True):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)

    subtrees = {}

    for token in doc:
        if output:
            print('{}:'.format(token.text))
        subtreeEls = []
        for els in token.subtree:
            if els != token:  # Avoid re-print the token under analysis
                subtreeEls.append(els)
                if output:
                    print('\t -> {}: {}'.format(els.text, els.dep_))
        if output:
            print('End subtree\n')
        subtreeEls.insert(0, token)
        subtrees[token.text] = subtreeEls
    return subtrees


def isSubtree(listOfTokens, refSentence):
    # Supposed having an ordered list of token: root -> dep 1 -> dep 2 -> ...

    subtrees = subtreeOfDependents(refSentence, False)

    if isinstance(listOfTokens[0], str):
        tempSentence = listOfTokens[0]
        if len(listOfTokens) > 1:
            for el in listOfTokens[1:]:
                tempSentence += ' {}'.format(el)
        nlp = spacy.load('en_core_web_sm')
        tempTokens = nlp(tempSentence)
        listOfTokens = tempTokens

    if listOfTokens[0].text in list(subtrees.keys()):
        index = 1  # Next token after subtree root
        for el in subtrees[listOfTokens[0].text][1:]:
            if el.text != listOfTokens[index].text:
                print('subtree does not fit: \'{}\' != \'{}\''
                      .format(el, listOfTokens[index]))
                return False  # Subtree differs by input one
            index += 1
        return True
    else:
        print('No subtree starting with \'{}\''.format(listOfTokens[0].text))
        return False  # No subtree with first element as root


def headOfSpan(listOfTokens):

    nlp = spacy.load('en_core_web_sm')
    if not isinstance(listOfTokens[0], str):  # List of spacy.tokens items
        wordsList = []
        for token in listOfTokens:
            wordsList.append(token.text)
        listOfTokens = wordsList

    sentence = listOfTokens[0]
    if len(listOfTokens) > 1:
        for el in listOfTokens[1:]:
            sentence += ' {}'.format(el)

    doc = nlp(sentence)
    span = doc[:]
    print('Head of: {} is \'{}\''.format(listOfTokens, span.root.text))

    return span.root


def objectsExtractor(sentence):
    depToFind = ['nsubj', 'dobj', 'dative']

    nlp = spacy.load('en_core_web_sm')

    doc = nlp(sentence)
    dictOfMatches = {}
    for el in depToFind:
        dictOfMatches[el] = []
    for token in doc:
        if token.dep_ in depToFind:
            tempList = []
            for el in token.subtree:
                tempList.append(el)
            dictOfMatches[token.dep_].append(tempList)
    return dictOfMatches


if __name__ == '__main__':
    sentence = 'I saw the man with a telescope'
    print(sentence)
    # paths = rootToTokenPath(sentence)  # Tested
    # subtrees = subtreeOfDependents(sentence)  # Tested
    # print(isSubtree(['with', 'a', 'telescope'], sentence))  # Tested
    # headOfSpan(sentence.split(' '))  # Tested
    # print(objectsExtractor('I saw the man'))  # Tested
