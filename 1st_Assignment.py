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
        # subtreeEls.insert(0, token)
        subtrees[token] = subtreeEls
    return subtrees


def isSubtree(listOfTokens, refSentence):
    # Supposed having an ordered list of token: root -> dep 1 -> dep 2 -> ...

    subtrees = subtreeOfDependents(refSentence, False)

    if not isinstance(listOfTokens[0], str):
        tempList = []
        for el in listOfTokens:
            tempList.append(el.text)
        listOfTokens = tempList

    # ListOfTokens now is a list of string tokens
    caselessList = []
    for el in listOfTokens:
        caselessList.append(el.casefold())
    listOfTokens = caselessList

    match = []
    for key in subtrees:
        if listOfTokens[0] == key.text:
            match.append(key)

    matchLists = []
    for key in match:
        stringList = []
        for val in subtrees[key]:
            stringList.append(val.text.casefold())
        stringList.insert(0, key.text.casefold())
        matchLists.append(stringList)

    existSubtree = False
    for _list in matchLists:
        if _list == listOfTokens:
            return True
    return existSubtree


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
    # print(isSubtree(['With', 'A', 'Telescope'], sentence))  # Tested
    # headOfSpan(sentence.split(' '))  # Tested
    # print(objectsExtractor('I saw the man'))  # Tested
