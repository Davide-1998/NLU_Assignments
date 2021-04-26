import spacy
import os
import conll


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
                tempNETags = tempNETags + '{} '.format(vec[3])

        sentence['text'].append(tempSentence.strip())
        sentence['POS_tag'].append(tempPOSTags.strip())
        sentence['SynChunkTag'].append(tempSyncChunksTags.strip())
        sentence['NE_tag'].append(tempNETags.strip())

    return sentence


def converter(token_ent, default='MISC'):
    converter = {'PERSON': 'PER',
                 'GPE': 'LOC',
                 'LOC': 'LOC',
                 'ORG': 'ORG',
                 'O': 'O'}
    return converter.get(token_ent, default)


def evaluateSpacy(conll_train, conll_test, max_sent=None, print_dicts=False):

    nlp = spacy.load('en_core_web_sm')

    test = loadConll(conll_test)

    if max_sent is not None and isinstance(max_sent, int):
        test_doc = list(nlp.pipe(test['text'][:max_sent]))
    else:
        test_doc = list(nlp.pipe(test['text']))
    print('Elements in doc format: {}'.format(len(test_doc)))

    # Retokenization to merge '-' elements (ex: dates, obj-obj)
    for doc in test_doc:
        with doc.retokenize() as retokenizer:
            index = 0
            startMerging = -1
            for token in doc:
                '''if token.ent_iob_ == 'B' and token.whitespace_ == '':
                    startMerging = index
                if (token.whitespace_ == ' ' and token.ent_iob_ == 'I'
                   and startMerging != -1) or \
                   (startMerging != -1 and index == len(doc)-1):

                    retokenizer.merge(doc[startMerging:index+1])
                    startMerging = -1'''

                if token.whitespace_ == '' and startMerging == -1:
                    startMerging = index
                if (token.whitespace_ == ' ' or index == len(doc)-1) \
                   and startMerging != -1:
                    retokenizer.merge(doc[startMerging:index+1])
                    startMerging = -1
                index += 1

    NE_dict_spacy = {}  # Dictionary to store spacy processed name entities
    for doc in test_doc:
        for token in doc:
            if token.ent_type_ == '':
                key = token.ent_iob_
            else:
                key = token.ent_iob_ + '-' + token.ent_type_
            if key not in NE_dict_spacy:
                NE_dict_spacy[key] = 1
            else:
                NE_dict_spacy[key] += 1

    NE_dict_conll = {}  # Dictionary to store conll NE divided by B and I
    for tag_list in test['NE_tag']:
        for tag in tag_list.split():
            if tag not in NE_dict_conll:
                NE_dict_conll[tag] = 1
            else:
                NE_dict_conll[tag] += 1

    grouped_NE_dict_conll = {}  # Dictionary to store conll NE not divided
    for key in NE_dict_conll:
        split_key = key
        if len(key) > 1:
            split_key = key.split('-')[1]
        if split_key not in grouped_NE_dict_conll:
            grouped_NE_dict_conll[split_key] = NE_dict_conll[key]
        else:
            grouped_NE_dict_conll[split_key] += NE_dict_conll[key]

    # Compute correct predictions

    correct_Prediction = {}
    doc_idx = 0
    for doc in test_doc:
        token_idx = 0
        for token in doc:
            if token.ent_type_ == '':
                key = token.ent_iob_
            else:
                key = token.ent_iob_ + '-' + \
                    converter(token.ent_type_, 'MISC')
            if key == test['NE_tag'][doc_idx].split()[token_idx]:
                if key not in correct_Prediction:
                    correct_Prediction[key] = 1
                else:
                    correct_Prediction[key] += 1
            token_idx += 1
        doc_idx += 1

    # Grouped NE Correct predictions
    grouped_correct_Prediction = {}
    for key in correct_Prediction:
        if len(key) > 1:
            splitted_key = key.split('-')[1]
        else:
            splitted_key = key

        if splitted_key not in grouped_correct_Prediction:
            grouped_correct_Prediction[splitted_key] \
                = correct_Prediction[key]
        else:
            grouped_correct_Prediction[splitted_key] \
                += correct_Prediction[key]

    if print_dicts:
        print('\nSpacy NE counts:')
        for key in NE_dict_spacy:
            print('Spacy NE: \'{}\' -> counts: {}'.format(key,
                                                          NE_dict_spacy[key]))

        print('\nConll NE counts:')
        for key in NE_dict_conll:
            print('Conll NE: \'{}\' -> counts: {}'.format(key,
                                                          NE_dict_conll[key]))

        print('\nGrouped Conll NE counts:')
        for key in grouped_NE_dict_conll:
            print('Grouped conll NE: \'{}\' -> counts: {}'
                  .format(key, grouped_NE_dict_conll[key]))

        print('\nCorrect prediction dictionary:')
        for key in correct_Prediction:
            print('\'Correct NE: {}\' -> counts: {}'
                  .format(key, correct_Prediction[key]))

        print('\nGrouped Correct prediction dictionary:')
        for key in grouped_correct_Prediction:
            print('\'Grouped NE: {}\' -> counts: {}'
                  .format(key, grouped_correct_Prediction[key]))

    # B & I accuracies
    print('\nB & I accuracies:')
    total_correct_tokens = 0
    total_tokens = 0
    for key in correct_Prediction:
        print('{} accuracy= {:0.4f}'
              .format(key, correct_Prediction[key]/NE_dict_conll[key]))
        total_correct_tokens += correct_Prediction[key]
        total_tokens += NE_dict_conll[key]
    print('total accuracy: {:0.4f}'.format(total_correct_tokens/total_tokens))

    # Grouped Accuracies
    print('\nGrouped Accuracies:')
    for key in grouped_correct_Prediction:
        print('{} accuracy= {:0.4f}'
              .format(key,
                      grouped_correct_Prediction[key] /
                      grouped_NE_dict_conll[key]))

    # Chunk accuracy (i.e entity accuracy)
    print('\nChunk accuracies:')
    spacyEnts = []
    for doc in test_doc:
        tempSpanList = []
        for span in doc.ents:
            tempSpanList.append(span)
        spacyEnts.append(tempSpanList)
    # print(spacyEnts)
    # for el in spacyEnts:
    #    for span in el:
    #        print(span.label_)

    # Try conll.evaluate:
    spacyEntList = []
    conllEntList = []

    for doc in test_doc[:max_sent]:
        tempList = []
        for ent in doc.ents:
            tempList.append([ent.text, converter(ent.label_, 'MISC')])
        spacyEntList.append(tempList)

    conllList = []
    sent_idx = 0
    for sent in test['text'][:max_sent]:
        token_idx = 0
        tempList = []
        for token in sent.split():
            NE_tag = test['NE_tag'][sent_idx].split()[token_idx]
            if NE_tag != 'O':
                tempList.append([token, NE_tag])
            token_idx += 1
        conllList.append(tempList)
        sent_idx += 1

    for sent in conllList:
        tempList = []
        idx = 0
        print(sent)
        tags = []
        for ent in sent:
            tags.append(ent[1].split('-')[0])
        Ilist = [tag for tag, val in enumerate(tags) if val == 'I']
        print(Ilist)


        '''if ent[1].split('-')[0] == 'B':
                if mergeStart == -1 or idx+1 == len(sent):
                    tempList.append([ent[0].strip(), ent[1].split('-')[1]])
                elif mergeEnd != -1:
                    txt = ''
                    for el in sent[mergeStart:mergeEnd+1]:
                        txt += el[0] + ' '
                    tempList.append([txt.strip(),
                                     sent[mergeStart][1].split('-')[1]])
                    mergeEnd = -1
                    mergeStart = -1
                mergeStart = idx

            if ent[1].split('-')[0] == 'I':
                if mergeEnd == -1:
                    mergeEnd = idx
                else:
                    mergeEnd += 1
                if idx + 1 == len(sent)-1:
                    txt = ''
                    for el in sent[mergeStart:mergeEnd+2]:
                        txt += el[0] + ' '
                    tempList.append([txt.strip(),
                                     sent[mergeStart][1].split('-')[1]])'''

        '''if ent[1].split('-')[0] == 'B':  # Begin of entity
                if mergeStart != -1 and mergeEnd != -1:
                    txt = ''
                    for el in sent[mergeStart:mergeEnd+1]:
                        txt += (el[0] + ' ')  # Add text of splices
                        print('New token: %s' % txt)
                    tempList.append([txt.stripe(),
                                     sent[mergeStart][1].split('-')[1]])
                    mergeStart = -1
                    mergeEnd = -1
                else:
                    mergeStart = idx
                    print('start index set to %d' % mergeStart)

            if ent[1].split('-')[0] == 'I':
                if mergeEnd == -1:
                    mergeEnd = idx
                    print('End index set to %d' % mergeEnd)
                else:
                    if mergeStart == -1:
                        mergeStart = idx - 1
                        print('Start index set to %d' % mergeStart)
                    mergeEnd += 1
                    print('End index increased to %d' % mergeEnd)'''
        idx += 1
        conllEntList.append(tempList)

    # print(spacyEntList, '\n')
    # print(conllList, '\n')
    # print('\n', conllEntList, '\n')

    '''for sent in test_doc[:max_sent]:
        tempList = []
        for token in sent:
            if token.ent_iob_ == 'O':
                tempList.append([token.text, token.ent_iob_])
            else:
                en = token.ent_iob_ + '-' + \
                     converter.get(token.ent_type_, 'MISC')
                tempList.append([token.text, en])
        spacyEntList.append(tempList)

    sent_idx = 0
    for sent in test['text'][:max_sent]:
        token_idx = 0
        tempList = []
        for token in sent.split():
            tempList.append([token,
                             test['NE_tag'][sent_idx].split()[token_idx]])
            token_idx += 1
        sent_idx += 1
        conllEntList.append(tempList)

    print(spacyEntList, '\n\n', conllEntList)'''

    '''conll_chunks = list(conll.get_chunks(os.getcwd() +
                                         '/data/conll2003/test.txt'))

    count = 0
    for el in spacyEntList:
        if el in conll_chunks:
            count += 1
    print(count)'''


def spacyEval_nlp_evaluate():
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


def frequentNE(ref):
    if isinstance(ref, str):  # Checks whether input is a sentence or Doc
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(ref)
    else:
        doc = ref  # Else is a spacy.tokens.Doc item

    ne_span_seq = []
    ents_seq = []
    for span in doc.noun_chunks:
        token_seq = []
        for token in span:
            if token.ent_type_ != '':
                token_seq.append(token.ent_type_)
        ne_span_seq.append([token_seq])

    for ents in doc.ents:
        ents_seq.append([ents.label_])

    unified_en_span_seq = []
    for seq in ne_span_seq:
        ent_in_span = []
        for span in seq:
            for ent in span:
                if ent not in ent_in_span:
                    ent_in_span.append(ent)
        unified_en_span_seq.append(ent_in_span)

    tempList = []
    for el in ents_seq:
        tempList.append(el[0])

    for span in unified_en_span_seq:
        for token in span:
            if token in tempList:
                tempList.remove(token)

    idx = 0
    for el in tempList:  # Residual elements
        idx = ents_seq.index([el])
        index = 0
        span_idx = 0
        stop = False
        if not stop:
            for span in unified_en_span_seq:
                if not stop:
                    for token in span:
                        if index != idx:
                            index += 1
                        else:
                            unified_en_span_seq.insert(span_idx, [el])
                            stop = True
                    span_idx += 1

    print(unified_en_span_seq)

    frequencies = {}

    for group in unified_en_span_seq:
        tempName = ''
        for el in group:
            tempName += '{} '.format(el)
        tempName = tempName.strip()

        if tempName not in frequencies:
            frequencies[tempName] = 1
        else:
            frequencies[tempName] += 1
    print(frequencies)


if __name__ == '__main__':
    # evaluateSpacy('train.txt', 'test.txt', 3, print_dicts=False)
    frequentNE('Apple\'s Steve Jobs died in 2011 in Palo Alto, California.')
