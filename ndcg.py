import math

import numpy as np

# compute the ndcg@k

def ndcg(tot_probs, tot_labels,k=5):

    total_ndcg=[]

    # get the queries
    
    for i in range(len(tot_probs)):
        
        # get the score
        #score = lrnk.score(q)
        
        probs = tot_probs[i]
        labels = tot_labels[i]
#        score = [ 0.99447438,  0.74102696,  0.86621875,  0.45964179,  0.96054857,
#        0.1436374 ,  0.73711684,  0.3826169 ,  0.38484938,  0.01030918]
#
#        # get the actual labels
#
#        #labels=q.get_labels()
#        
#        labels = [0,1,0,0,0,0,0,0,1]

        # sort labels for the ideal dcg

        sorted_labels_=[i for i,j in enumerate(labels) if j==1 ]

        # rank the documents
        #print(sorted_labels_)
        sorted_scores=[i[0] for i in sorted(enumerate(probs), key=lambda x:x[1],reverse=True)]
        #print(sorted_scores)
        # keep the labels for the first k documents

        sorted_labels=[i for i in sorted_labels_ if i in sorted_scores[:k]]
        #print(sorted_labels)
        # compute ndcg@k
        
        dcg = 0
        #print(k,len(labels))

        for n,r in enumerate(sorted_scores[:min(k,len(labels))]):

            #print(r)

            dcg+=((2**labels[r])-1)/(math.log(n+2,2))

            

        perf_dcg = 0

        for n,r in enumerate(sorted_labels_[:k]):

            

            perf_dcg+=((2**labels[r])-1)/(math.log(n+2,2))
            

            

        
        ndcg=0.0

        if perf_dcg!=0:

            ndcg=dcg/perf_dcg

        total_ndcg.append(ndcg)

        
    #print(total_ndcg)    

    return np.mean(np.array(total_ndcg))
    
  
    
#probs = [[ 0.99447438,  0.74102696,  0.86621875,  0.45964179,  0.96054857,
#        0.1436374 ,  0.73711684,  0.3826169 ,  0.05,  0.01030918],[ 0.99447438,  0.74102696,  0.86621875,  0.45964179]]
#
#        # get the actual labels
#
#        #labels=q.get_labels()
#        
#labels = [[0],[1,0]]

#probs = [[0.4480540156364441, 0.3782067894935608, 0.3119354546070099, 0.274630069732666, 0.268954336643219, 0.2422584593296051, 0.2323857992887497, 0.21957191824913025, 0.21462120115756989, 0.22041276097297668],
#         [0.3744608163833618, 0.34171533584594727, 0.31242072582244873, 0.31378762221336365, 0.30822455883026123, 0.30192509293556213, 0.30804523825645447, 0.2676906883716583, 0.2422294169664383, 0.22545841336250305]]
#labels = [[1, 0, 1, 1, 1, 0, 0, 1, 0, 1],[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]]
#ndcg_ = ndcg(probs, labels,3)
#print(ndcg_)