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


def find_alignment_score(a1, a2, w_inp):
    nodes1 = a1.nodes
    nodes2 = a2.nodes
    n1_keys = sorted(nodes1.keys())
    n2_keys = sorted(nodes2.keys())
    
    posc_dict = {}
    min_matches = 0
    for k1 in n1_keys:
        try:
            posc_dict[nodes1[k1].pos] = (posc_dict[nodes1[k1].pos][0] + 1, posc_dict[nodes1[k1].pos][1]) 
        except KeyError:
            posc_dict[nodes1[k1].pos] = (1, 0)
    for k2 in n2_keys:
        try:
            posc_dict[nodes2[k2].pos] = (posc_dict[nodes2[k2].pos][0], posc_dict[nodes2[k2].pos][1] + 1) 
        except KeyError:
            posc_dict[nodes2[k2].pos] = (0, 1)
    
    for k1 in posc_dict.keys():
        min_matches += min(posc_dict[k1])
    
    m = Model("mip1")
        
    var_dict = {}
    fvs_dict = {}
    obj = 0
    i = 0
    n_tot = 0
    for n1 in n1_keys:
        n_constr = 0
        j = 0
        for n2 in n2_keys:
            #Add an ILP variable only if node types match
            if nodes1[n1].pos == nodes2[n2].pos and nodes1[n1].n_type != 'root' and nodes2[n2].n_type != 'root':
                #Create node-pair ILP variable and get node-pair feature vector
                v_name = "n_" + str(i) + '_' + str(j)
                v_t = m.addVar(vtype=GRB.BINARY, name=v_name)
                n_fv = ilp_feature_extraction.get_node_fv(n1, a1, n2, a2)
                var_dict[v_name] = v_t
                fvs_dict[v_name] = n_fv
                n_tot += v_t
                
                #print nodes1[n1].word, nodes2[n2].word, nodes1[n1].polarity, nodes2[n2].polarity, np.dot(n_fv, w_inp[:len(n_fv)])
                #Add the node-pair variable to global objective
                obj += v_t * np.dot(n_fv, w_inp[:len(n_fv)])
            j += 1
        i += 1
    if np.linalg.norm(w_inp) == 0:
        m.addConstr(n_tot >= min_matches, "node_tot")
    
    for i in range(len(n1_keys)):
        constr = 0
        is_blank = True
        for j in range(len(n2_keys)):
            try:
                v_name = "n_" + str(i) + '_' + str(j)
                constr += var_dict[v_name]                
                is_blank = False
            except KeyError:
                continue
        if not is_blank:
            #Each node in nodes1 much match only to one node at max
            m.addConstr(constr <= 1, "nc_1_" + str(i))
            
    for j in range(len(n2_keys)):
        constr = 0
        is_blank = True
        for i in range(len(n1_keys)):
            try:
                v_name = "n_" + str(i) + '_' + str(j)
                constr += var_dict[v_name]
                is_blank = False
            except KeyError:
                continue
        if not is_blank:
            #Each node in nodes2 much match only to one node at max
            m.addConstr(constr <= 1, "nc_2_" + str(j))
        
    i = 0
    for n1 in n1_keys:
        j = 0
        n1_edges = nodes1[n1].outgoing
        for n2 in n2_keys:
            n2_edges = nodes2[n2].outgoing
            nv1_name = "n_" + str(i) + '_' + str(j)
            try:
                nv_1 = var_dict[nv1_name]
            except KeyError:
                j += 1
                continue
            for e1 in n1_edges:
                nv_1_n2 = n1_keys.index(e1[0])
                e_obj = 0
                for e2 in n2_edges:
                    nv_2_n2 = n2_keys.index(e2[0])
                    nv2_name = "n_" + str(nv_1_n2) + '_' + str(nv_2_n2)
                    try:
                        nv_2 = var_dict[nv2_name]
                    except KeyError:
                        continue
    
                    #Creating edge-pair ILP variable
                    e_name = "e_" + str(i) + '_' + str(nv_1_n2) + '_' + str(j) + '_' + str(nv_2_n2)
                    e_t = m.addVar(vtype=GRB.BINARY, name=e_name)
                    e_fv = ilp_feature_extraction.get_edge_fv((nodes1[n1], nodes1[e1[0]], e1[1]), a1, (nodes2[n2], nodes2[e2[0]], e2[1]), a2)
                    var_dict[e_name] = e_t
                    fvs_dict[e_name] = e_fv
                    e_obj += e_t
                    
                    #Adding edge-pair ILP variable to global objective
                    obj += e_t * (np.dot(e_fv, w_inp[len(n_fv):]))
                    
                    #If edge matches, corresponding nodes must match and vice-versa
                    m.addConstr(e_t <= nv_1, "ec_" + str(i) + '_' + str(nv_1_n2) + '_' + str(j) + '_' + str(nv_2_n2) + '_1')
                    m.addConstr(e_t <= nv_2, "ec_" + str(i) + '_' + str(nv_1_n2) + '_' + str(j) + '_' + str(nv_2_n2) + '_2')
                    m.addConstr(nv_1 + nv_2 - 1 <= e_t, "ec_" + str(i) + '_' + str(nv_1_n2) + '_' + str(j) + '_' + str(nv_2_n2) + '_3')
                #m.addConstr(e_obj >= 1, "edge_tot_" + str(nv_1_n2))
            j += 1
        i += 1
    
    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam('OutputFlag', False)
    m.optimize()
    
    aligned_fdict = {}
    for v in m.getVars():
        aligned_fdict[v.varName] = (fvs_dict[v.varName], v.x)
    
    return (m.objVal, aligned_fdict)
