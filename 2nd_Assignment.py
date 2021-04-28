import spacy
import os
import conll
import pandas as pd
from sklearn.metrics import classification_report as report


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
    # print('Elements in doc format: {}'.format(len(test_doc)))

    # Retokenization to merge '-' elements (ex: dates, obj-obj)
    for doc in test_doc:
        with doc.retokenize() as retokenizer:
            index = 0
            startMerging = -1
            for token in doc:
                if token.whitespace_ == '' and startMerging == -1:
                    startMerging = index
                if (token.whitespace_ == ' ' or index == len(doc)-1) \
                   and startMerging != -1:
                    retokenizer.merge(doc[startMerging:index+1])
                    startMerging = -1
                index += 1

    NE_dict_spacy = {}  # Dictionary to store spacy processed name entities
    doc_spacy_test_list = []  # -> New
    for doc in test_doc:
        token_list = []  # -> New
        for token in doc:
            if token.ent_type_ == '':
                key = token.ent_iob_
            else:
                key = token.ent_iob_ + '-' + token.ent_type_
            doc_spacy_test_list.append(converter(key))  # -> New
            if key not in NE_dict_spacy:
                NE_dict_spacy[key] = 1
            else:
                NE_dict_spacy[key] += 1
        # doc_spacy_test_list.append(token_list)  # -> New

    NE_dict_conll = {}  # Dictionary to store conll NE divided by B and I
    doc_conll_test_list = []  # -> New
    for tag_list in test['NE_tag']:
        t = []  # -> New
        for tag in tag_list.split():
            doc_conll_test_list.append(tag)  # -> New
            if tag not in NE_dict_conll:
                NE_dict_conll[tag] = 1
            else:
                NE_dict_conll[tag] += 1
        # doc_conll_test_list.append(t)  # -> New

    # ##################################################### NEW
    scores = report(doc_conll_test_list, doc_spacy_test_list,\
                    output_dict= True, zero_division=0)
    print('Accuracy on spacy prediction: {:0.4f}'.format(scores['accuracy']))
    # ##################################################### NEW

    '''grouped_NE_dict_conll = {}  # Dictionary to store conll NE not divided
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
                      grouped_NE_dict_conll[key]))'''

    # Chunk accuracy (i.e entity accuracy)
    sent_idx = 0
    ref_list = []
    hyp_list = []
    for sent in test['text'][:max_sent]:
        token_idx = 0
        ref_token_list = []
        hyp_token_list = []
        for token in sent.split():
            ref_token_list.append(
                    [token,
                     test['NE_tag'][sent_idx].split()[token_idx]])

            if test_doc[sent_idx][token_idx].ent_type_ == '':
                hyp_token_list.append(
                    [test_doc[sent_idx][token_idx].text,
                     test_doc[sent_idx][token_idx].ent_iob_])
            else:
                hyp_token_list.append(
                    [test_doc[sent_idx][token_idx].text,
                     test_doc[sent_idx][token_idx].ent_iob_ +
                     '-' + converter(
                        test_doc[sent_idx][token_idx].ent_type_)])

            token_idx += 1
        ref_list.append(ref_token_list)
        hyp_list.append(hyp_token_list)
        sent_idx += 1

    measures = conll.evaluate(ref_list, hyp_list)
    # Make fancy table:
    measureShow = pd.DataFrame().from_dict(measures, orient='index')
    print(measureShow.round(decimals=3))


def groups_NE(ref):
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

    return unified_en_span_seq


def frequencies_comp(listOfSentences, num_print=5, print_all=False):
    frequencies = {}
    for sent in listOfSentences:
        for group in sent:
            tempName = ''
            for el in group:
                tempName += '{} '.format(el)
            tempName = tempName.strip()

            if tempName != '':
                if tempName not in frequencies:
                    frequencies[tempName] = 1
                else:
                    frequencies[tempName] += 1

    # sorting in descending order
    sorted_keys = sorted(frequencies, key=frequencies.__getitem__, \
                         reverse=True)
    sorted_frequencies = {}
    for keys in sorted_keys:
        sorted_frequencies[keys] = frequencies.get(keys)

    if not print_all:
        i = 0
        while i < num_print:
            print('\'{}\': {}'.format(sorted_keys[i], \
                                      sorted_frequencies[sorted_keys[i]]))
            i += 1
    else:
        for key in sorted_frequencies:
            print('\'{}\': {}'.format(key, sorted_frequencies[key]))

    return sorted_frequencies


def computeConllFreqs(conllFile):
    conllFile = loadConll('test.txt')['text']
    nlp = spacy.load('en_core_web_sm')
    sents = list(nlp.pipe(conllFile))
    sent = []
    for doc in sents:
        sent.append(groups_NE(doc))
    frequencies_comp(sent)


def postProcess(text):
    # print(text)

    if isinstance(text, str):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
    else:
        doc = text

    token_list = []
    for token in doc:  # Create list of tokens -> indirectly index of doc
        token_list.append(token.text)

    print('Starting ents:')
    for el in doc.ents:
        print([el])

    for token in doc:
        # print([token.text, token.ent_iob_+'-'+token.ent_type_, token.dep_])
        ent_type = token.ent_iob_
        if ent_type != 'O':
            ent_type += '-' + token.ent_type_

        if token.dep_ == 'compound':
            # print(token.text)
            head = token.head  # Other token in compound relation
            lists_of_entities = doc.ents

            in_span = False
            for span in lists_of_entities:
                if head in span:
                    found_span = span
                    if token in found_span:
                        print('Both elements of compound in span')
                        in_span = True
                    else:
                        token_idx = token_list.index(token.text)
                        head_idx = token_list.index(head.text)
                        start_span_idx = token_list.index(span[0])
                        with doc.retokenize() as retokenizer:
                            if token_idx != head_idx:
                                if token_idx > head_idx:
                                    retokenizer.merge(
                                            doc[start_span_idx:token_idx+1])
                                else:
                                    retokenizer.merge(
                                            doc[start_span_idx:head_idx+1])
                        in_span = True

            if not in_span:  # No span contains them
                with doc.retokenize() as retokenizer:
                    token_idx = token_list.index(token.text)
                    head_idx = token_list.index(head.text)
                    if token_idx != head.idx:
                        if token_idx < head_idx:
                            print(token_idx, head_idx)
                            retokenizer.merge(doc[token_idx:head_idx+1])
                        else:
                            retokenizer.merge(doc[head_idx:token_idx+1])

    print('\nProcessed ents')
    for el in doc.ents:
        print([el])

    return doc


    '''for span in lists_of_entities:
                if head in span:
                    if token in span:
                        print('Both element of compound in span')
                        continue
                    else:
                        index_compound = token_list.index(token.text)
                        head_index = token_list.index(head.text)
                        start_index = token_list.index(span[0].text)
                        if head_index < index_compound:
                            doc.set_ents([spacy.tokens.Span(doc,
                                                            start_index,
                                                            index_compound+1,
                                                            head.ent_type_)])
                        else:
                            doc.set_ents([spacy.tokens.Span(doc,
                                                            start_index,
                                                            head_index+1,
                                                            head.ent_type_)])
                else:
                    head_index = token_list.index(head.text)
                    index_compound = token_list.index(token.text)
                    if head_index < index_compound:
                        doc.set_ents([spacy.tokens.Span(doc,
                                                        head_index,
                                                        index_compound+1,
                                                        ent_type)])
                    else:
                        doc.set_ents([spacy.tokens.Span(doc,
                                                        index_compound,
                                                        head_index+1,
                                                        ent_type)])
                list_of_doc.append(doc)
    new_doc = spacy.tokens.Doc.from_docs(list_of_doc)'''


    '''token_list = []
    for span in doc.ents:
        print([span])
        for token in span:
            if token.dep_ == 'compound':
                child = token.children
                for t in child:
                    print('Child: ', t.text, t.dep_, t.ent_type_)
            if token.ent_iob_ != 'O':
                token_list.append((token.text, token.ent_iob_ + '-' + \
                                   token.ent_type_))
            else:
                token_list.append((token.text, token.ent_iob_))
    print('\n', token_list)'''


if __name__ == '__main__':
    # evaluateSpacy('train.txt', 'test.txt', print_dicts=False)
    # frequencies in conll:
    # computeConllFreqs('test.txt')

    # postProcess('Apple\'s Steve Jobs died in 2011 in Palo Alto, California.')

    sentence = 'Soccer - Japan get lucky win , china in surprise defeat'
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    postProcess(doc)
