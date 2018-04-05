import nltk
import xml.etree.ElementTree as ET
import pickle
import text_utils


filepath = '/homes/rpujari/scratch/data/semeval-t11/'

gen_sem = ['accompanier', 'age', 'beneficiary', 'cause', 'compared-to', 'concession', 'condition',\
          'consist-of', 'degree', 'destination', 'direction', 'domain', 'duration', 'employed-by',\
          'example', 'extent', 'frequency', 'instrument', 'li', 'location', 'manner', 'medium',\
          'mod', 'mode', 'name', 'part', 'path', 'polarity', 'poss', 'purpose', 'source', 'subvert',\
          'subset', 'time', 'topic', 'value']

quant = ['quant', 'unit', 'scale']

date_ent = ['day', 'month', 'year', 'weekday', 'time', 'timezone', 'quarter', 'dayperiod', 'season',\
           'year2', 'decade', 'century', 'era']

list_rel = ['op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8', 'op9', 'op10']

frame_arg = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']


whQW = [u'what', u'who', u'how', u'which', u'whose']

whQW_fw = [u'when', u'where', u'whom', u'why']

beQW = [u'is', u'are', u'was', u'were']

doQW = [u'do', u'does', u'did', u'has', u'have', u'had']

modals = [u'can', u'will', u'shall', u'could', u'would', u'should']


def read_data(fp, data_answers = []):
    ret_data = []
    data = ET.parse(fp)
    droot = data.getroot()
    i = 0
    a_num = 0
    for inst in droot:
        p_id = inst.attrib['id']
        para = inst[0].text
        p_doc = []
        p_sents = nltk.sent_tokenize(para)
        for p_sent in p_sents:
            if type(p_sent) == type('string'):
                p_doc.append(text_utils.spacy_nlp(unicode(p_sent, errors='ignore')))
            else:
                p_doc.append(text_utils.spacy_nlp(p_sent))
        qas = inst[1]
        q_pairs = []
        j = 0
        for qa in qas:
            q = qa.attrib['text']
            q_type = qa.attrib['type']
            if type(q) == type('string'):
                q_doc = text_utils.spacy_nlp(unicode(q, errors='ignore'))
            else:
                q_doc = text_utils.spacy_nlp(q)
            answers = []
            k = 0
            for ans in qa:
                if data_answers == []:
                    a = ans.attrib['text']
                else:
                    a = data_answers[a_num]
                if type(a) == type('string'):
                    a_doc = text_utils.spacy_nlp(unicode(a, errors='ignore'))
                else:
                    a_doc = text_utils.spacy_nlp(a)
                a_tv = ans.attrib['correct']
                answers.append((a_doc, a_tv))
                k += 1
                a_num += 1
            q_pairs.append((q_doc, answers, q_type))
            j += 1
        ret_data.append((p_doc, q_pairs))
        i += 1
    return ret_data




with open(filepath + 'script_id/dev_script_dict.pkl', 'rb') as infile:
    dev_script_dict = pickle.load(infile)
    
with open(filepath + 'script_id/dev_script_info.pkl', 'rb') as infile:
    dev_script_info = pickle.load(infile)

with open(filepath + 'train_qw_dict.pkl', 'rb') as infile:
    train_qw_dict = pickle.load(infile)
    
with open(filepath + 'dev_qw_dict.pkl', 'rb') as infile:
    dev_qw_dict = pickle.load(infile)

with open(filepath + 'answer_amrs/train_ans_text/all_answers_new.txt', 'r') as infile:
    train_answers = infile.read().split('\n')[:-1]

with open(filepath + 'answer_amrs/dev_ans_text/all_answers_new.txt', 'r') as infile:
    dev_answers = infile.read().split('\n')[:-1]

train_data = read_data(filepath + 'train-data.xml', train_answers)

dev_data = read_data(filepath + 'dev-data.xml', dev_answers)

train_text_dict = {}
for key in train_qw_dict.keys():
    qw_list = train_qw_dict[key]
    for p_num, q_num in qw_list:
        if train_data[p_num][1][q_num][2] == 'text':
            try:
                train_text_dict[key].append((p_num, q_num))
            except KeyError:
                train_text_dict[key] = [(p_num, q_num)]

dev_text_dict = {}
for key in dev_qw_dict.keys():
    qw_list = dev_qw_dict[key]
    for p_num, q_num in qw_list:
        if dev_data[p_num][1][q_num][2] == 'text':
            try:
                dev_text_dict[key].append((p_num, q_num))
            except KeyError:
                dev_text_dict[key] = [(p_num, q_num)]




