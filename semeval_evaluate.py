import pickle
import math
import numpy as np
from datetime import datetime
import copy

import amr_utils
import text_utils
import text_feature_extraction
import semeval_utils
import semeval_imports
import ilp_feature_extraction
import semeval_inference

def evaluate_qset(data, q_list, w_inp, qw):
    #print str(datetime.now())
    n_corr = 0.0
    tot_q = 0.0
    text_qc = 0
    cs_qc = 0
    text_tot = 0
    cs_tot = 0
    for p_num, q_num in q_list:
        para = data[p_num]
        question, answers, q_type = para[1][q_num] 
        t_gfp = semeval_utils.filepath + 'dev_amrs/amr_' + str(p_num) + '.txt'
        corr_idx = -1
        corr_comb_score = -float('inf')
        max_idx = -1
        max_comb_score = -float('inf')
        max_al_score = -float('inf')
        corr_al_score = -float('inf')
        k = 0
        print p_num, q_num
        print q_type
        print question.text
        for ans in answers:
            a_gfp = semeval_utils.filepath + 'answer_amrs/dev_ans_amrs/ans_new_' + str(p_num) + '_' + str(q_num) + '_' + str(k)
            save_fp = semeval_utils.filepath + 'dev_alignments/' + qw + '/'
            script_afp = None
            print ans[0].text
            try:
                s_name = semeval_utils.dev_script_info[p_num]
                s_fp = semeval_utils.filepath + 'script_amrs/amr_' + '-'.join(s_name.split()) + '-0'
                f_lines = open(s_fp, 'r').read()
                script_afp = s_fp
            except IOError:
                pass#print s_name
            comb_score, al_score, best_comb, comb_fv, align_dict = semeval_inference.inference(para, p_num, q_num, k, w_inp, t_gfp, a_gfp, save_fp, script_afp=None)
            if ans[1] == 'True':
                corr_idx = k
                corr_comb_score = comb_score
                corr_al_score = al_score
            elif (comb_score + al_score) > (max_comb_score + max_al_score):
                max_comb_score = comb_score
                max_al_score = al_score
                max_idx = k
            k += 1
        corr_score, max_score, score_decomp = ilp_feature_extraction.noramlize_scores(corr_comb_score, max_comb_score, corr_al_score, max_al_score)
        print max_idx, corr_idx, max_score, corr_score
        print ''

        if corr_score > max_score:
            #dev_res_dict[str(p_num) + '_' + str(q_num)] = 'correct'
            n_corr += 1
            if q_type == 'text':
                text_qc += 1
            elif q_type == 'commonsense':
                cs_qc += 1
        else:
            pass#dev_res_dict[str(p_num) + '_' + str(q_num)] = 'wrong'
        
        tot_q += 1
        if q_type == 'text':
            text_tot += 1
        elif q_type == 'commonsense':
            cs_tot += 1
    #print str(datetime.now())
    print qw, text_qc, text_tot, cs_qc, cs_tot
    return text_qc, text_tot, cs_qc, cs_tot

