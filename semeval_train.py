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


def train(data, w_init, q_set, save_w, save_fp, epochs = 1, eta = 0.1, gamma = 0.0001):
    #Start time
    print str(datetime.now())
    
    #Initializations
    w_train = copy.deepcopy(w_init)
    
    cum_delta_w = np.array([0.0 for x in w_train])
    
    for epoch in range(epochs):
        q_num = 0
        n_corr = 0.0
        i = 0
        for para in data:
            if i >= 0 and i <= 1500:
                text_gfp = semeval_utils.filepath + 'train_amrs/amr_' + str(i) + '.txt'
                q_pairs = para[1]

                j = 0
                for question, answers, q_type in q_pairs:
                    if (i, j) in q_set or len(q_set) == 0: 
                        corr_idx = -1
                        corr_comb_score = -float('inf')
                        max_idx = -1
                        max_comb_score = -float('inf')
                        max_al_score = -float('inf')
                        corr_al_score = -float('inf')
                        best_res = []
                        corr_res = []
                        k = 0

                        print i, j
                        print question.text
                        for ans in answers:
                            a_gfp = semeval_utils.filepath + 'answer_amrs/train_ans_amrs/ans_new_'  + str(i) + '_' + str(j) + '_' + str(k)
                            comb_score, al_score, best_comb, comb_fv, align_dict = semeval_inference.inference(para, i, j, k, w_train, text_gfp, a_gfp, save_fp)
                            print ans[0].text
                            if ans[1] == 'True':
                                corr_idx = k
                                corr_comb_score = comb_score
                                corr_al_score = al_score
                                correct_res = (comb_score, best_comb, comb_fv, align_dict)
                            elif (comb_score + al_score) > (max_comb_score + max_al_score):
                                max_comb_score = comb_score
                                max_al_score = al_score
                                max_idx = k
                                best_res = (comb_score, best_comb, comb_fv, align_dict)
                            k += 1
                        corr_score, max_score, score_decomp = ilp_feature_extraction.noramlize_scores(corr_comb_score, max_comb_score, corr_al_score, max_al_score) 
                        print max_idx, corr_idx, max_score, corr_score

                        if score_decomp[2] <= score_decomp[3] + 0.1:
                            if corr_score > max_score:
                                print 'Marginal correct'
                            else:
                                print 'Incorrect'
                            delta_w = np.array([0.0 for x in w_train])
                            
                            comb_fv = correct_res[2]
                            align_fv_dict = correct_res[3]
                            align_keys = align_fv_dict.keys()
#                             delta_w[:len(comb_fv)] += comb_fv
                            for key in align_keys:
                                if 'n' in key:
                                    if align_fv_dict[key][1] == 1:
                                        n_fv = align_fv_dict[key][0]
                                        delta_w[len(comb_fv):len(comb_fv) + len(n_fv)] += n_fv
                                elif 'e' in key:
                                    if align_fv_dict[key][1] == 1:
                                        e_fv = align_fv_dict[key][0] 
                                        delta_w[-len(e_fv):] += e_fv
                            
                            comb_fv = best_res[2]
                            align_fv_dict = best_res[3]
                            align_keys = align_fv_dict.keys()
#                             delta_w[:len(comb_fv)] -= comb_fv
                            for key in align_keys:
                                if 'n' in key:
                                    if align_fv_dict[key][1] == 1:
                                        n_fv = align_fv_dict[key][0]
                                        delta_w[len(comb_fv):len(comb_fv) + len(n_fv)] -= n_fv
                                elif 'e' in key:
                                    if align_fv_dict[key][1] == 1:
                                        e_fv = align_fv_dict[key][0]
                                        delta_w[-len(e_fv):] -= e_fv
            
            
                            print delta_w
                            cum_delta_w += eta * delta_w
                            w_train[len(comb_fv):] += eta * delta_w[len(comb_fv):]
                
                            print w_train

                            #Save w_train
                            with open(save_w, 'wb') as outfile:
                                pickle.dump(w_train, outfile)

                        else:
                            print 'Safely correct'
                            #w_train[len(comb_fv):] = [((1 - gamma) * x) for x in w_train[len(comb_fv):]]
                            n_corr += 1
                        print ''
                        q_num += 1
                    j += 1
            i += 1
        print n_corr, q_num
    print str(datetime.now())
    return w_train, (n_corr / q_num), cum_delta_w


