# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:55:32 2017

@author: theft
"""


import pickle
import gzip
import json

from collections import Counter

def read_file(path = 'YandexRelPredChallenge.txt'):
    yandex_log = []
    with open(path, 'r') as f:
        yandex_log = f.readlines()

    yandex_log_ = []
    for line in yandex_log:
        line = line.strip()
        yandex_log_.append(line.split('\t'))
        
    return yandex_log_





def trainq2dict(file_lines):
    
    relevance_dict = {}


    for line in file_lines:
    
        if line[0] not in relevance_dict:
            relevance_dict[line[0]] = {'relevant':[],'irrelevant':[]}
            
        if int(line[3]) == 1:

            relevance_dict[line[0]]['relevant'].append(int(line[2]))
        else:
            relevance_dict[line[0]]['irrelevant'].append(int(line[2]))
            
    return relevance_dict
        
def values2counter(relevance_dict):
    
    for key, value in relevance_dict.items():

        relevance_dict[key]['irrelevant'] = Counter(value['irrelevant']) 
        relevance_dict[key]['relevant'] = Counter(value['relevant']) 
        
    return relevance_dict


def validate_rel(relevance_dict):
    
    

    for key in relevance_dict.copy().keys():
    
    
        for doc in relevance_dict[key]['relevant'].copy().keys():

            if doc in relevance_dict[key]['irrelevant']:

                if relevance_dict[key]['relevant'][doc] > relevance_dict[key]['irrelevant'][doc]:
                    del relevance_dict[key]['irrelevant'][doc]
                    
                elif relevance_dict[key]['relevant'][doc] < relevance_dict[key]['irrelevant'][doc]:
                    del relevance_dict[key]['relevant'][doc]
                    
                else:
                    del relevance_dict[key]['irrelevant'][doc] , relevance_dict[key]['relevant'][doc]
            
        relevance_dict[key]['relevant'] = list(relevance_dict[key]['relevant'].keys())
        relevance_dict[key]['irrelevant'] = list(relevance_dict[key]['irrelevant'].keys())
            
    return relevance_dict


def filter_rel(relevance_dict):
    
    for key in relevance_dict.copy().keys():

        if len(relevance_dict[key]['relevant'])<1:
            del relevance_dict[key]
    
    return relevance_dict


def pickle_dict(dictionary, file_name):
    
    
    with open(file_name,'wb') as f:
        pickle.dump(dictionary, f)
        
        
def ndcg_iterate_over_gzip(input_file, relevance_dict, q2d={}, num_of_lines=1e6):
    
    gzip_file = gzip.open(input_file,'r')
    
    queries_eval = set()
    

    
    
    # loop over the input_file and append the lines in a list
    for i,line in enumerate(gzip_file):
        
        query_id = json.loads(line.decode("utf-8").strip("\n"))['query_id']

        if query_id not in q2d:
            q2d[query_id] = set()
            q2d[query_id].add(tuple(json.loads(line.decode("utf-8").strip("\n"))['document_ids']))
        else:
            q2d[query_id].add(tuple(json.loads(line.decode("utf-8").strip("\n"))['document_ids']))
            
        if (i+1)%1e5==0 and i>0:
            
            #print("\rRead the "+str(i+1)+" lines",end="")
            print("\rProgress "+str(int((i+1)/num_of_lines*100))+" %",end="")
            
          
    print()
    
    return queries_eval, q2d

    
def get_ndcg_queries_over_gzip(input_file, relevance_dict, q2d={}, num_of_lines=1e6):
    
    gzip_file = gzip.open(input_file,'r')
    
    queries_eval = set()
    
    
    
    # loop over the input_file and append the lines in a list
    for i,line in enumerate(gzip_file):
        
        query_id = json.loads(line.decode("utf-8").strip("\n"))['query_id']
        if str(query_id) in relevance_dict.keys() :
            
            
            queries_eval.add(query_id)
            docs = json.loads(line.decode("utf-8").strip("\n"))['document_ids']

             
            if query_id not in q2d or tuple(docs) not in q2d[query_id]: 
                
                for doc in docs:
                
                
                    
                    if doc in relevance_dict[str(query_id)]['relevant']:
                        
                        
                                
                        
                            
                            with open('dataset/ndcg'+str(input_file)[-4]+'.test', 'a') as outfile:
                                json.dump(json.loads(line.decode("utf-8")), outfile)
                                outfile.write('\n')
                            
                                
                    if query_id not in q2d:
                        q2d[query_id] = set()
                        q2d[query_id].add(tuple(docs))
                    else:
                        q2d[query_id].add(tuple(docs))
                        
                    break
                

        if (i+1)%1e5==0 and i>0:
            
            #print("\rRead the "+str(i+1)+" lines",end="")
            print("\rProgress "+str(int((i+1)/num_of_lines*100))+"%",end="")
            
    print()
    
    return queries_eval, q2d
    
    
    
    
#path = 'dataset/Trainq.txt'
##
#lines = read_file(path)
##
###print(len(lines))
##
#relevance = trainq2dict(lines)
##
#relevance = values2counter(relevance)
##
###print(relevance['357'])
##
#relevance = validate_rel(relevance)
##
#relevance = filter_rel(relevance)
#
##print(len(relevance))
#
##print(relevance['357'])
#
##pickle_rel(relevance, "dataset/Trainq")
#
#ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
#
#for i in range(8):
#    
#    if i<4:
#        file = 'dataset/train_query_sessions_with_behavioral_features.part_'+str(i)+'.gz'
#    else:
#        file = 'dataset/test_query_sessions_with_behavioral_features.part_'+str(i-4)+'.gz'
#    print("Start Iterating over the {:s} the Input File ".format(ordinal(i+1)))
#    x_,y_ = ndcg_iterate_over_gzip(file,_)
#    
#print("Finished")
