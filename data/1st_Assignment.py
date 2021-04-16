import spacy


def rootToTokenPath(sentence):
    # Finds and prints dependency path from Root to specified token

    nlp = spacy.load('en_core_web_sm')  # Create language loading a pipline
    doc = nlp(sentence)  # Create doc object by feeding the input
                         # Sentence to the previous defined pipeline

    for token in doc:
        listOfDependencies = []
        while token.dep_ != 'ROOT':
            listOfDependencies.insert(0, (token.text, token.dep_))
            token = token.head
        listOfDependencies.insert(0, (token.text, token.dep_))

        for el in listOfDependencies:
            print(' -[{}]-> {}'.format(el[1], el[0]), end='')
        print('\n')


if __name__ == '__main__':
    rootToTokenPath('He is the king with the rotten crown')
