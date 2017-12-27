import math

import numpy as np

# compute the ndcg@k

def ndcg(tot_probs, tot_labels,k=5):

    total_ndcg=[]

    
    
    for i in range(len(tot_probs)):
        
        # get the probabilities
        probs = tot_probs[i]

        # get the labels
        labels = tot_labels[i]


        # sort labels for the ideal dcg

        sorted_labels_=[i for i,j in enumerate(labels) if j==1 ]

        # rank the documents
        #print(sorted_labels_)
        sorted_scores=[i[0] for i in sorted(enumerate(probs), key=lambda x:x[1],reverse=True)]
        
        # keep the labels for the first k documents
        sorted_labels=[i for i in sorted_labels_ if i in sorted_scores[:k]]

        # compute ndcg@k
        
        dcg = 0

        for n,r in enumerate(sorted_scores[:min(k,len(labels))]):


            dcg+=((2**labels[r])-1)/(math.log(n+2,2))

            

        perf_dcg = 0

        for n,r in enumerate(sorted_labels_[:k]):

            

            perf_dcg+=((2**labels[r])-1)/(math.log(n+2,2))
            

            

        
        ndcg=0.0

        if perf_dcg!=0:

            ndcg=dcg/perf_dcg

        total_ndcg.append(ndcg)

        

    return np.mean(np.array(total_ndcg))
    
  
    
