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


def evaluateSpacy(conll_test, max_sent=None, print_dicts=False):

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

    doc_spacy_test_list = []
    for doc in test_doc:
        for token in doc:
            if token.ent_type_ == '':
                key = token.ent_iob_
            else:
                key = token.ent_iob_ + '-' + token.ent_type_
            doc_spacy_test_list.append(converter(key))

    doc_conll_test_list = []
    for tag_list in test['NE_tag']:
        for tag in tag_list.split():
            doc_conll_test_list.append(tag)

    scores = report(doc_conll_test_list, doc_spacy_test_list,
                    output_dict=True, zero_division=0)
    print('Accuracy on spacy prediction: {:0.4f}\n'.format(scores['accuracy']))

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

    ne_span_seq = []  # Sequence of Named Entities span coming from Doc
    for span in doc.noun_chunks:
        token_seq = []
        for token in span:
            if token.ent_type_ != '':
                token_seq.append(token.ent_type_)
        ne_span_seq.append([token_seq])

    ents_seq = []  # Sequence of entities
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
    sorted_keys = sorted(frequencies, key=frequencies.__getitem__,
                         reverse=True)
    sorted_frequencies = {}
    for keys in sorted_keys:
        sorted_frequencies[keys] = frequencies.get(keys)

    if not print_all:
        i = 0
        for key in sorted_frequencies:
            if len(key.split()) > 1 and i < num_print:
                print('\'{}\': {}'.format(key,
                                          sorted_frequencies[key]))
                i += 1

    else:
        for key in sorted_frequencies:
            print(key.split(), len(key.split()))
            print('\'{}\': {}'.format(key, sorted_frequencies[key]))

    print('Keys in dictionary', len(sorted_frequencies))

    return sorted_frequencies


def computeConllFreqs(conllFile, max_num=5, print_all=False):
    conllFile = loadConll('test.txt')['text']
    nlp = spacy.load('en_core_web_sm')
    sents = list(nlp.pipe(conllFile))
    sent = []
    for doc in sents:
        sent.append(groups_NE(doc))
    frequencies_comp(sent, max_num, print_all)


def postProcess(text):
    print('input text: ', text)
    if isinstance(text, str):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
    else:
        doc = text

    print('\nStarting entities:')
    for el in doc.ents:
        print([el])

    comp_dict = {}

    for token in doc:
        print([token.text, token.ent_iob_, token.ent_type_])

    print('\nElements in compound relation:')
    for token in doc:
        if token.dep_ == 'compound':
            print([token.text])
            if token.head not in comp_dict:
                comp_dict[token.head] = [token]
            else:
                comp_dict[token.head].insert(token.i, token)

    for key in comp_dict:
        if key.i > comp_dict[key][-1].i:
            comp_dict[key].append(key)

    total_els = 0
    for key in comp_dict:
        total_els += len(comp_dict[key])

    composed_ents = []
    temp_ent = []
    index = 0
    for key in comp_dict:
        for token in comp_dict[key]:
            if len(temp_ent) == 0:
                temp_ent.append(comp_dict[key][0])
            elif token != temp_ent[-1]:
                if token not in temp_ent and token.i - temp_ent[-1].i == 1:
                    temp_ent.append(token)
                else:
                    composed_ents.append(temp_ent)
                    temp_ent = [token]
            if index == total_els-1:  # Definetly last
                composed_ents.append(temp_ent)
            index += 1

    prev_ents = list(doc.ents)
    copy_ents = list(doc.ents)
    insert_ref = []
    for ent in prev_ents:
        in_ents = False
        for token in ent:
            for el in composed_ents:
                if token in el:
                    in_ents = True
                    insert_idx = prev_ents.index(ent)
                    ref_idx = composed_ents.index(el)
                    if [insert_idx, ref_idx] not in insert_ref:
                        insert_ref.append([insert_idx, ref_idx])
        if in_ents:
            copy_ents.remove(ent)

    new_ents = []
    for el in copy_ents:
        new_ents.append(list(el))

    for el in insert_ref:
        if el[0] < len(new_ents) and el[1] < len(composed_ents):
            new_ents.insert(el[0], composed_ents[el[1]])
            composed_ents.remove(composed_ents[el[1]])

    if composed_ents != []:  # Append the grouped entities that are left
        for el in composed_ents:
            new_ents.append(el)

    span_ents = []
    for el in new_ents:  # Converts grouped entities into  span
        start_idx = el[0].i
        end_idx = el[-1].i
        if doc[start_idx].ent_iob_ != '':
            iob = doc[start_idx].ent_iob_ + '-' + doc[start_idx].ent_type_
        else:
            iob = doc[start_idx].ent_iob_
        span = spacy.tokens.Span(doc, start_idx, end_idx+1, label=iob)
        span_ents.append(span)

    doc.ents = span_ents  # Overwrites previous doc entities
    print('\nNew entities:')
    for el in list(doc.ents):
        print([el])

    return doc


if __name__ == '__main__':
    # evaluateSpacy('test.txt', print_dicts=False)
    # frequencies in conll:
    computeConllFreqs('test.txt')

    # postProcess('Apple\'s Steve Jobs died in 2011 in Palo Alto, California.')
    #postProcess('He said a proposal last month by EU Farm Commissioner Franz'+
    #            ' Fischler to ban sheep brains')
    '''sentence = 'Soccer - Japan get lucky win , china in surprise defeat'
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    postProcess(doc)'''
