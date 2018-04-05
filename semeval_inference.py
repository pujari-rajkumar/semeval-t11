import pickle
import math
import numpy as np
from gurobipy import *
from datetime import datetime
import copy

import amr_utils
import text_utils
import text_feature_extraction
import semeval_utils
import semeval_imports
import ilp_feature_extraction
import ilp_alignment

def inference(para, p_idx, q_idx, a_idx, w_inp, text_gfp, ans_gfp, save_fp, comb_len=2, num_candidates=5, script_afp=None):
    ques = para[1][q_idx]
    p_len = len(para[0])
    
    #print p_len
    
    #Create combinations of sentences
    para_combs = ilp_feature_extraction.get_combs(p_len, comb_len)
    #print para_combs
    
    #Initializations
    q_doc = ques[0]
    ic_dict = text_feature_extraction.create_ic_dict(para[0])
    a = ques[1][a_idx]
    a_doc = a[0]
    comb_scores = []
    align_scores = []
    sent_scores = []
    sent_fvs = []
    
    #Reading answer and question AMRs
    ans_amr = amr_utils.read_amr(ans_gfp, [0])
    
    #Calculating combination scores
    for i, text_doc in zip(range(len(para[0])), para[0]):
        #fv1 = extract_features_1(q_doc, text_doc)
        z_amr = amr_utils.read_amr(text_gfp, [i])
        fv2 = text_feature_extraction.extract_features_2(a_doc, ans_amr, q_doc, text_doc, z_amr, ic_dict)
        sent_score = np.dot(w_inp[:len(fv2)], np.array(fv2))
        sent_fvs.append(np.array(fv2))
        sent_scores.append(sent_score)
        
    if script_afp:
        script_amr = amr_utils.read_amr(script_afp, range(100))
        z_amr = amr_utils.read_amr(text_gfp, range(100)) 
        text_eas = []
        text_eas += text_feature_extraction.ent_act_amr(z_amr)
        script_acts = get_script_acts(script_amr.docs)
        script_subj = None
        for t_ea in text_eas:
            if t_ea[1].lemma_ in script_acts and t_ea[0].lemma_ != 'entity':
                script_subj = t_ea[0]
                break
        script_combs = ilp_feature_extraction.get_combs(len(script_amr.text), comb_len)
        script_ic_dict = text_feature_extraction.create_ic_dict(script_amr.docs)
        script_sent_scores = []
        script_sent_fvs = []
        for script_doc in script_amr.docs:
            fv2 = text_feature_extraction.extract_features_2(a_doc, q_doc, script_doc, script_ic_dict)
            script_sent_score = np.dot(w_inp[:len(fv2)], np.array(fv2))
            script_sent_fvs.append(np.array(fv2))
            script_sent_scores.append(script_sent_score)
        for comb in script_combs:
            ss_list = [script_sent_scores[x] for x in comb]
            comb_fvs = [script_sent_fvs[x] for x in comb]
            comb_fv = np.mean(np.array(comb_fvs), axis=0)
            z_score = np.mean(ss_list)
            comb_scores.append((comb, comb_fv, z_score, len(para_combs)))
            
        
    for comb in para_combs:
        ss_list = [sent_scores[x] for x in comb]
        comb_fvs = [sent_fvs[x] for x in comb]
        comb_fv = np.mean(np.array(comb_fvs), axis=0)
        z_score = np.mean(ss_list)
        comb_scores.append((comb, comb_fv, z_score, 0))
        
    #print len(comb_scores)
        
    sorted_comb_scores = sorted(comb_scores, key=lambda x:x[2], reverse=True)
    
    #Computing alignment scores for top-k combinations
    l_sc = len(sorted_comb_scores)
    n_cand = min(num_candidates, l_sc)
    
    for i in range(n_cand):
        if sorted_comb_scores[i][3] == 0:
            text_amr = amr_utils.read_amr(text_gfp, sorted_comb_scores[i][0])
#             try:
#                 run_coref(text_amr.nodes, text_amr.text)
#             except:
#                 ;
        else:
            text_amr = amr_utils.read_amr(script_afp, sorted_comb_scores[i][0])
            if script_subj:
                add_script_subj(text_amr, script_subj)
        alignment = ilp_alignment.find_alignment_score(ans_amr, text_amr, w_inp[len(fv2):])
        align_scores.append((i, alignment, text_amr, sorted_comb_scores[i][2] + alignment[0]))
        
    #Sorting alignments based on sum of alignment and combination scores
    sorted_align_scores = sorted(align_scores, key=lambda x:x[3], reverse=True)
        
    #Saving TEXT graph, ANSWER graph and ALIGNMENTS to file
    align_fp = save_fp + 'align_' + ilp_feature_extraction.get_ans_id(p_idx, q_idx, a_idx) + '.txt'
    align_writer = open(align_fp, 'w')
    ans_dot = ans_amr.convert_to_dot('ANS')
    text_dot = sorted_align_scores[0][2].convert_to_dot('TEXT')
    align_writer.write(ans_dot)
    align_writer.write(text_dot)
    best_alignment_dict = sorted_align_scores[0][1][1]
    a_keys = best_alignment_dict.keys()
    align_writer.write('Alignment:\n')
    for a_key in a_keys:
        if best_alignment_dict[a_key][1] == 1:
            align_writer.write(a_key + '\n')
    align_writer.close()
    
    comb_idx = sorted_align_scores[0][0]
    high_comb_score = sorted_comb_scores[comb_idx][2]
    high_align_score = sorted_align_scores[0][1][0]
    
    return (high_comb_score, high_align_score, sorted_comb_scores[comb_idx][0], sorted_comb_scores[comb_idx][1], best_alignment_dict)
