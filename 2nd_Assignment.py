import spacy
from spacy.training import Example
import os
import conll


def NE_tag_conversion(NE_tag):
    if '-' in NE_tag:
        NE_tag = NE_tag.split('-')[1]

    conversion = {'PER': 'PERSON',
                  'LOC': 'GPE',
                  'O': 'DATE',
                  'MISC': 'PRODUCT'}
    return conversion.get(NE_tag, NE_tag)


def loadConll(conllFile):
    pathToConllFile = conllFile

    if os.sep not in conllFile:  # Checks if is a path to the file
        pathToConllFile = os.getcwd() + os.sep + 'data' + os.sep \
            + 'conll2003' + os.sep + conllFile

    corpus = conll.read_corpus_conll(pathToConllFile)
    corpus = corpus[1:]  # Removes DOCSTART line
    print('Elements in corpus: {}'.format(len(corpus)))
    sentence = {'text': [],
                'POS_tag': [],
                'SynChunkTag': [],
                'NE_tag': []}
    for _list in corpus:
        tempSentence = ''
        tempPOSTags = ''
        tempSyncChunksTags = ''
        tempNETags = ''
        for vec in _list:
            vec = vec[0].split(' ')
            if '-DOCSTART-' not in vec:
                tempSentence = tempSentence + '{} '.format(vec[0])
                tempPOSTags = tempPOSTags + '{} '.format(vec[1])
                tempSyncChunksTags = tempSyncChunksTags + '{} '.format(vec[2])
                tempNETags = tempNETags + '{} '.format(
                    NE_tag_conversion(vec[3]))

        sentence['text'].append(tempSentence.strip())
        sentence['POS_tag'].append(tempPOSTags.strip())
        sentence['SynChunkTag'].append(tempSyncChunksTags.strip())
        sentence['NE_tag'].append(tempNETags.strip())

    return sentence


def evaluateSpacy(conll_train, conll_test, overwriteDoc=False):

    nlp = spacy.load('en_core_web_sm')

    test = loadConll(conll_test)
    test_doc = list(nlp.pipe(test['text']))

    for doc in test_doc:
        # Retokenization to merge '-' elements (ex dates)
        with doc.retokenize() as retokenizer:
            index = 0
            startMerging = -1
            for token in doc:
                if token.ent_iob_ == 'B' and token.whitespace_ == '':
                    startMerging = index
                if (token.whitespace_ == ' ' and token.ent_iob_ == 'I'
                   and startMerging != -1) or \
                   (startMerging != -1 and index == len(doc)-1):

                    retokenizer.merge(doc[startMerging:index+1])
                    startMerging = -1
                index += 1

    '''NE_dict_spacy = {}  # Dictionary to store spacy processed name entities
    for doc in test_doc:
        # Accuracy is number of correct prediction/ total prediction
        for token in doc:
            if token.ent_type_ not in NE_dict_spacy:
                NE_dict_spacy[token.ent_type_] = 1
            else:
                NE_dict_spacy[token.ent_type_] += 1

    for key in NE_dict_spacy:
        print('NE: \'{}\' -> counts: {}'.format(key, NE_dict_spacy[key]))
    '''

    '''pathToConllFile = os.getcwd() + os.sep + 'data' + os.sep \
                      + 'conll2003' + os.sep +'test.txt'
    corpus = conll.read_corpus_conll(pathToConllFile)
    tempCorpus = []
    for el in corpus:
        tempList = []
        for token in el:
            token = token[0].split()
            tempList.append(token)
        tempCorpus.append(tempList)
    corpus = tempCorpus
    conEval = conll.conlleval(corpus)
    for el in conEval:
        print(el, conEval[el])'''

    '''idx = 0
    listExamples = []
    print('Building list of Spacy Examples...')
    for txt in test_doc:  # Iterate on a list of doc items
        nlp_example = Example.from_dict(
                                     txt,
                                     {'words': txt.text.split(),
                                      'pos': test['POS_tag'][idx].split(),
                                      'tags': test['SynChunkTag'][idx].split()}
                                        )
        listExamples.append(nlp_example)
        idx += 1

    print('Elements as examples: {}'
          .format(len(listExamples)))

    score = nlp.evaluate(listExamples)
    for key in score:
        print('{} : {}'.format(key, score[key]))'''


if __name__ == '__main__':
    evaluateSpacy('train.txt', 'test.txt')
