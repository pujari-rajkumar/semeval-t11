import nltk
import pickle
import math
import numpy as np
import xml.etree.ElementTree as ET
import os
import jsonrpc
from simplejson import loads
from nltk.tree import Tree
import codecs
import itertools
from itertools import product
from nltk.corpus import wordnet as wn
from gurobipy import *
from datetime import datetime
import copy
import random
from gensim.models import KeyedVectors
import amr_utils
import text_utils
import semeval_utils


def combine_docs(d_list):
    if len(d_list) == 1:
        return d_list[0]
    text = []
    for doc in d_list:
        text.append(doc.text)
    comb_str = ' '.join(text)
    comb_doc = text_utils.spacy_nlp(comb_str)
    return comb_doc

def get_ans_id(p_idx, q_idx, a_begin):
    a_idx = str(p_idx) + '_' + str(q_idx) + '_' + str(a_begin)
    return a_idx

def get_combs(l, m):
    ret_list = []
    i = 1
    while i <= m:
        for comb in itertools.combinations(range(l), i):
            ret_list.append(list(comb))
        i += 1
    return ret_list

def edit_dist(str1, str2, m , n):
    lev_table = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        lev_table[i][0] = i
    for j in range(n + 1):
        lev_table[0][j] = j
    for i in range(m):
        for j in range(n):
            if str1[i] == str2[j]:
                lev_table[i + 1][j + 1] = lev_table[i][j]
            else:
                s1 = lev_table[i][j + 1]
                s2 = lev_table[i + 1][j]
                s3 = lev_table[i][j]
                lev_table[i + 1][j + 1] = min(s1, s2, s3) + 1
    return lev_table[m][n]

def get_all_synsets(word, pos=None):
    for ss in wn.synsets(word):
        for lemma in ss.lemma_names():
            yield (lemma, ss.name())

def get_all_hyponyms(word, pos=None):
    for ss in wn.synsets(word, pos=pos):
            for hyp in ss.hyponyms():
                for lemma in hyp.lemma_names():
                    yield (lemma, hyp.name())

def get_all_hypernyms(word, pos=None):
    for ss in wn.synsets(word, pos=pos):
            for hyp in ss.hypernyms():
                for lemma in hyp.lemma_names():
                    yield (lemma, hyp.name())

def get_all_similar_tos(word, pos=None):
    for ss in wn.synsets(word):
            for sim in ss.similar_tos():
                for lemma in sim.lemma_names():
                    yield (lemma, sim.name())

def get_all_antonyms(word, pos=None):
    for ss in wn.synsets(word, pos=None):
        for sslema in ss.lemmas():
            for antlemma in sslema.antonyms():
                    yield (antlemma.name(), antlemma.synset().name())

def get_all_also_sees(word, pos=None):
        for ss in wn.synsets(word):
            for also in ss.also_sees():
                for lemma in also.lemma_names():
                    yield (lemma, also.name())

def get_all_synonyms(word, pos=None):
    for x in get_all_synsets(word, pos):
        yield (x[0], x[1], 'ss')
    for x in get_all_hyponyms(word, pos):
        yield (x[0], x[1], 'hyp')
    for x in get_all_hypernyms(word, pos):
        yield (x[0], x[1], 'hyp')
    for x in get_all_similar_tos(word, pos):
        yield (x[0], x[1], 'sim')
    for x in get_all_antonyms(word, pos):
        yield (x[0], x[1], 'ant')
    for x in get_all_also_sees(word, pos):
        yield (x[0], x[1], 'also')

def add_script_subj(script_amr, subj):
    subj_node = amr_node(subj.text.encode('utf-8'), 'NOUN', 'ARG', (-1, -1), '1', '1')
    script_amr.nodes['1'] = subj_node
    root_node = script_amr.nodes['0']
    for edge in root_node.outgoing:
        script_amr.nodes[edge[0]].add_outgoing('1', 'ARG0')

def get_script_acts(s_docs):
    s_acts = []
    doc_keys = s_docs.keys()
    for doc_key in doc_keys:
        doc = s_docs[doc_key]
        for tok in doc:
            if tok.head == tok:
                s_acts.append(tok.lemma_)
    return s_acts

def noramlize_scores(corr_cs, max_cs, corr_as, max_as):
    cs_max = max(abs(corr_cs), abs(max_cs))
    as_max = max(abs(corr_as), abs(max_as))
    corr_cs_norm = 0
    max_cs_norm = 0
    corr_as_norm = 0
    max_as_norm = 0
    if cs_max > 0 and as_max > 0:
        corr_s = (corr_cs / cs_max) + (corr_as / as_max)
        max_s = (max_cs / cs_max) + (max_as / as_max)
        corr_cs_norm = corr_cs / cs_max
        max_cs_norm = max_cs / cs_max
        corr_as_norm = corr_as / as_max
        max_as_norm = max_as / as_max
    elif cs_max > 0:
        corr_s = corr_cs / cs_max
        max_s = max_cs / cs_max
        corr_cs_norm = corr_cs / cs_max
        max_cs_norm = max_cs / cs_max
    elif as_max > 0:
        corr_s = corr_as / as_max
        max_s = max_as / as_max
        corr_as_norm = corr_as / as_max
        max_as_norm = max_as / as_max
    else:
        corr_s = 0
        max_s = 0
    #corr_s = corr_cs + corr_as
    #max_s = max_cs + max_as
    return corr_s, max_s, [corr_cs_norm, max_cs_norm, corr_as_norm, max_as_norm]

def get_node_wsim(n1, a1, n2, a2):
    n1_w = a1.nodes[n1].word
    n2_w = a2.nodes[n2].word
    if n1_w.lower() == u'i':
        n1_w = u'narrator'
    if n2_w.lower() == u'i':
        n2_w = u'narrator'
    wv_sim1 = text_utils.get_wv_similarity(n1_w, n2_w)
    wn_sim1 = text_utils.get_wn_similarity(n1_w, n2_w)
    n1_doc = a1.docs[int(n1.split('.')[0])]
    #print a2.docs
    n2_doc = a2.docs[int(n2.split('.')[0])]
    n1_w = n1_doc[a1.nodes[n1].span_id[0]].text
    n2_w = n2_doc[a2.nodes[n2].span_id[0]].text
    if n1_w.lower() == u'i':
        n1_w = u'narrator'
    if n2_w.lower() == u'i':
        n2_w = u'narrator'
    wv_sim2 = text_utils.get_wv_similarity(n1_w, n2_w)
    wn_sim2 = text_utils.get_wn_similarity(n1_w, n2_w)
    wv_sim = max(wv_sim1, wv_sim2)
    wn_sim = max(wn_sim1, wn_sim2)
    pol = a1.nodes[n1].polarity * a2.nodes[n2].polarity
    if pol == -1:
        wv_sim = (1 - wv_sim) / 2
        wn_sim = (1 - wn_sim) / 2
    return wv_sim, wn_sim

def get_wn_realtions(n1, a1, n2, a2):
    n1_doc = a1.docs[int(n1.split('.')[0])]
    n2_doc = a2.docs[int(n2.split('.')[0])]
    n1_w = n1_doc[a1.nodes[n1].span_id[0]].text
    n2_w = n2_doc[a2.nodes[n2].span_id[0]].text
    if n1_w.lower() == u'i':
        n1_w = u'narrator'
    if n2_w.lower() == u'i':
        n2_w = u'narrator'
    
    is_hyp = 0
    is_sim = 0
    is_ant = 0

    for x in get_all_synonyms(n1_w):
        if n2_w == x[0].strip():
            if x[2] == 'ss' or x[2] == 'sim':
                is_sim = 1
            if x[2] == 'hyp':
                is_hyp = 1
            if x[2] == 'ant':
                is_ant = 1
    
    return is_hyp, is_sim, is_ant

def get_node_edit_dist(n1, a1, n2, a2):
    n1_doc = a1.docs[int(n1.split('.')[0])]
    n2_doc = a2.docs[int(n2.split('.')[0])]
    n1_lem = n1_doc[a1.nodes[n1].span_id[0]].lemma_
    n2_lem = n2_doc[a2.nodes[n2].span_id[0]].lemma_
    return edit_dist(n1_lem, n2_lem, len(n1_lem), len(n2_lem))

def get_node_og_match(n1, a1, n2, a2):
    n1_es = set([x[1] for x in a1.nodes[n1].outgoing])
    n2_es = set([x[1] for x in a2.nodes[n2].outgoing])
    c = 0.0
    for n1_e in n1_es:
        if n1_e in n2_es:
            c += 1
    if len(n1_es) == 0:
        return c, 0
    return c, (c / len(n1_es))

def get_node_fv(n1, a1, n2, a2):
    wv_sim, wn_sim = get_node_wsim(n1, a1, n2, a2)
    is_hyp, is_sim, is_ant = get_wn_realtions(n1, a1, n2, a2)
    e_dist = 1 / (get_node_edit_dist(n1, a1, n2, a2) + 1)
    og_m, og_fc = get_node_og_match(n1, a1, n2, a2)
    pol = a1.nodes[n1].polarity * a2.nodes[n2].polarity
    return np.array([wv_sim, wn_sim, e_dist, og_m, og_fc, is_hyp, is_sim, is_ant, pol])

def get_edge_fv(e1, a1, e2, a2):
    fv = []
    e1_s = e1[0]
    e1_d = e1[1]
    e2_s = e2[0]
    e2_d = e2[1]
    
    e1_s_w = e1_s.word
    e1_d_w = e1_d.word
    e2_s_w = e2_s.word
    e2_d_w = e2_d.word
    
    wv1_sum = text_utils.get_wv(e1_s_w) + text_utils.get_wv(e1_d_w)
    wv2_sum = text_utils.get_wv(e2_s_w) + text_utils.get_wv(e2_d_w)
    
    wv1_diff = text_utils.get_wv(e1_s_w) - text_utils.get_wv(e1_d_w)
    wv2_diff = text_utils.get_wv(e2_s_w) - text_utils.get_wv(e2_d_w)
    
    fv.append(text_utils.cos_similarity(wv1_sum, wv2_sum))
    fv.append(text_utils.cos_similarity(wv1_diff, wv2_diff))
    
    if e1[2] in semeval_utils.gen_sem and e2[2] in semeval_utils.gen_sem:
        fv.append(1)
    elif e1[2] in semeval_utils.quant and e2[2] in semeval_utils.quant:
        fv.append(1)
    elif e1[2] in semeval_utils.date_ent and e2[2] in semeval_utils.date_ent:
        fv.append(1)
    elif e1[2] in semeval_utils.list_rel and e2[2] in semeval_utils.list_rel:
        fv.append(1)
    elif e1[2] in semeval_utils.frame_arg and e2[2] in semeval_utils.frame_arg:
        fv.append(1)
    else:
        fv.append(0)
        
    if e1[2] == e2[2]:
        fv.append(1)
    else:
        fv.append(0)
        
    return np.array(fv)


