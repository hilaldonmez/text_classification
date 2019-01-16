import re
import stemmer
import os
from collections import defaultdict
import math

full_file_path="Dataset"
stopwords_path="stopwords.txt" 
ref_topics={"earn":[] , "acq":[] , "money-fx":[], "grain":[], "crude":[] }
test_topics={"earn":[] , "acq":[] , "money-fx":[], "grain":[], "crude":[] }

#%%
# Generate stoplist
def fill_stopword(stopwords_path):    
    stoplist=[]
    file = open(stopwords_path, 'r') 
    for line in file: 
        token=re.split('\n',line)
        stoplist.append(token[0].strip()) 
    file.close()
    return stoplist
    
#%%
# Tokenize the text    
def tokenize_text(txt,stoplist): 
    txt = re.sub('&.*?;', '', txt)
    vocabulary=re.findall(r'(?ms)\W*(\w+)', txt)
    vocabulary=[s.lower() for s in vocabulary if s.lower() not in stoplist and len(s)>1 and s.isalpha()]
    return vocabulary    

#%%
def read_text(test_path,news_body_train,news_body_test,ref_topics,stoplist,test_topics):
    
    doc_store=[] 
    with open(test_path,encoding='iso-8859-1') as file:  
        fulltext = file.read() 
        doc_store=re.split('</REUTERS>', fulltext, flags=re.IGNORECASE)
    file.close()
    for each_doc in range(len(doc_store)):
        if "<TOPICS><D>" in doc_store[each_doc]:
            index_topic=re.search("<TOPICS><D>", doc_store[each_doc]).end()
            index_topic_end=re.search("</D></TOPICS>", doc_store[each_doc]).start()
            topic=doc_store[each_doc][index_topic:index_topic_end]
            topics=re.split('</D><D>', topic, flags=re.IGNORECASE)
            once_in=False
            real_topic=''
            for each_topic in topics:
                if each_topic in ref_topics:
                    if once_in:
                        once_in=False
                        continue
                    else:
                        real_topic=each_topic
                        once_in=True
            
            if once_in and "NEWID" in doc_store[each_doc] :
                train=True
                train_test_index=re.search(r'LEWISSPLIT=(.*)CGISPLIT', doc_store[each_doc])
                
                index=re.search("NEWID", doc_store[each_doc]).end()+2
                indexforinner=[m.start() for m in re.finditer(r">",doc_store[each_doc][index:])][0]
                docID=doc_store[each_doc][index:index+indexforinner-1]
                docID=int(docID)
                check=False
                if "TEST"  in train_test_index.group(0):
                    train=False
                    check=True
                    test_topics[real_topic]+=[docID]
                elif "TRAIN"  in train_test_index.group(0):
                    check=True
                    ref_topics[real_topic]+=[docID]
                if check:
                    if '<TITLE>' not in doc_store[each_doc]:
                         news_title=[]
                    else:            
                        index_title=re.search("<TITLE>", doc_store[each_doc]).end()
                        index_title_end=re.search("</TITLE>", doc_store[each_doc]).start()                
                        news_title=tokenize_text(doc_store[each_doc][index_title:index_title_end],stoplist)
                    if '<BODY>' not in doc_store[each_doc]:
                        if train:
                            news_body_train[docID]=news_title
                        else:    
                            news_body_test[docID]=news_title 
                    else: 
                        doc_store[each_doc]=re.split('<BODY>', doc_store[each_doc], flags=re.IGNORECASE)[1]
                        doc_store[each_doc]=re.split('</BODY>', doc_store[each_doc], flags=re.IGNORECASE)[0]
                        text=news_title+tokenize_text(doc_store[each_doc],stoplist)
                        if train:    
                            news_body_train[docID]=text
                        else:
                            news_body_test[docID]=text
                            
    return news_body_train,news_body_test,ref_topics,test_topics    

#%%    
# final preprocessing for  train set    
def final_preprocessing(news_body,ref_topics): 
    distinct_stem=defaultdict(lambda: 'Vanilla') 
    positional={}
    stem_doc={}
    for each_class in ref_topics:
        positional[each_class]={}
        for docID in ref_topics[each_class]:
            for each_word in news_body[docID]:
                stem=stemmer.stemming(each_word)
                distinct_stem[stem]=docID
                if stem not in positional[each_class]:
                    positional[each_class][stem]=1
                else:
                    positional[each_class][stem]+=1
                if stem not in stem_doc:
                    stem_doc[stem]=[docID]
                elif docID not in stem_doc[stem]: 
                    stem_doc[stem].append(docID)
                 
    return positional,distinct_stem,stem_doc   

#%%
# final preprocessing for test set    
def final_preprocessing_test_doc(news_body_test):
    stem_freq={}
    for each_doc in news_body_test:
        stem_freq[each_doc]={}
        for each_word in news_body_test[each_doc]:
            stem=stemmer.stemming(each_word)
            if stem not in stem_freq[each_doc]:
                stem_freq[each_doc][stem]=1
            else:
                stem_freq[each_doc][stem]+=1

    return stem_freq            

#%%
def read_all_documents(full_file_path,ref_topics,stoplist,test_topics): 
    news_body_train={}
    news_body_test={}                           
    for filename in os.listdir(full_file_path):  
        if '.sgm' in filename:
            news_body_train,news_body_test,ref_topics,test_topics=read_text(full_file_path+"/"+filename,news_body_train,news_body_test,ref_topics,stoplist,test_topics)
            
    return news_body_train,news_body_test,ref_topics,test_topics        
    
#%%
# Calculate the total stem in the train set
def find_total_stem_num(positional):
    stem_num={}
    for each_class in positional:
        stem_num[each_class]=sum(positional[each_class].values())
    return  stem_num

#%%
# Calculate the total number of stem in the train set from mutual information     
def find_total_stem_num_mf(positional,mutual_features):
    stem_num={}
    for each_class in mutual_features:
        stem_num[each_class]=0
        for each_stem in mutual_features[each_class]:
            stem_num[each_class]+=positional[each_class][each_stem]
    return  stem_num
     
#%%   
# Calculate the probability for each class in the train set     
def get_prob_each_class(ref_topics,total_doc_num):
    prob_class=[]
    for each_class in ref_topics:
        #print(each_class)
        prob_class.append(len(ref_topics[each_class])/total_doc_num)
        
    return prob_class    
#%%
# Naive Bayes for each test document
# return the class label for this document    

def multinomial_naive_bayes(test_doc,ref_topics,positional,news_body_train,distinct_stems_number,prob_class,stem_num,features):    
    
    final_prob=[math.log10(prob) for prob in prob_class.copy()]
    for each_stem in test_doc:
        count=0
        for each_class in ref_topics:
            if each_stem not in features[each_class]:
                inner=0
            else:
                inner=positional[each_class][each_stem]
            prob_word=(inner+1)/(distinct_stems_number+stem_num[each_class])                
            final_prob[count]+=math.log10(prob_word)*test_doc[each_stem]      
            count+=1
    
    max_val = max(final_prob)
    max_idx = final_prob.index(max_val)
    l=list(ref_topics.keys())
    #print(l[max_idx])
    return l[max_idx]    

    
       
#%%
# final classification of naive bayes and naive bayes with mutual information features    
def final_classification(query,test_stem,ref_topics,positional,news_body_train,distinct_stems_number,prob_class,mutual_features):
    final_test_class={"earn":[] , "acq":[] , "money-fx":[], "grain":[], "crude":[] }
    if int(query)==1:
        stem_num=find_total_stem_num(positional)    
        for each_test_doc in test_stem: 
            final_test_class[multinomial_naive_bayes(test_stem[each_test_doc],ref_topics,positional,news_body_train,distinct_stems_number,prob_class,stem_num,positional)]+=[each_test_doc]
        
    else:
        stem_num=find_total_stem_num_mf(positional,mutual_features)
        for each_test_doc in test_stem:      
            final_test_class[multinomial_naive_bayes(test_stem[each_test_doc],ref_topics,positional,news_body_train,distinct_stems_number,prob_class,stem_num,mutual_features)]+=[each_test_doc]
    
    return final_test_class    


#%%
def list_intersection(list1,list2):
    temp=set(list1)
    temp=list(temp.intersection(set(list2))) 
    return temp            
#%%
# Select features with mutual information    
def mutual_information(ref_topics,positional,stem_doc,total_train_doc_num):
    
    distinct_stem=defaultdict(lambda: 'Vanilla')
    poss_stem={"earn":[] , "acq":[] , "money-fx":[], "grain":[], "crude":[] }
    for each_class in positional:
        stem_class={}
        for each_stem in positional[each_class]:
            n_double=[]
            n_single=[]
            n_11=len(list_intersection(ref_topics[each_class],stem_doc[each_stem]))
            n_10=len(stem_doc[each_stem])-n_11
            n_01=len(ref_topics[each_class])-n_11
            n_00=total_train_doc_num-(n_11+n_10+n_01)
            n_double.append(n_00)
            n_double.append(n_01)
            n_double.append(n_11)
            n_double.append(n_10)
            
            for i in range(len(n_double)):
                if n_double[i]==0:
                    n_double[i]=1    
            
            #silinecek
            n1=n_10+n_11
            n0=n_00+n_01
            n_1=n_11+n_01
            n_0=n_00+n_10
            n_single.append(n_0)
            n_single.append(n0)
            n_single.append(n_1)
            n_single.append(n1)
            n_single.append(n_0)
        
            I=0
            for i in range(len(n_double)):
                if n_double[i]!=0:
                    I+=n_double[i]*math.log2((total_train_doc_num*n_double[i])/(n_single[i]*n_single[i+1]))
            I/=total_train_doc_num    
            stem_class[each_stem]=I 
            
        b=sorted(stem_class, key=stem_class.get,reverse=True)
        #print(b[:50])
        
        poss_stem[each_class]=b[:50]    
        for i in range(50):
            distinct_stem[b[i]]=1
            
        
    return poss_stem,distinct_stem       
#%%
def macro_average_score(test_topics,final_test_class):
    av_precision=0
    av_recall=0
    for each_class in test_topics:
        
        precision=len(list_intersection(test_topics[each_class],final_test_class[each_class]))/len(final_test_class[each_class])
        recall=len(list_intersection(test_topics[each_class],final_test_class[each_class]))/len(test_topics[each_class])        
        f_measure=(2*precision*recall)/(precision+recall)
        print(each_class," p: ",precision," r: ",recall,' f: ',f_measure )
        av_precision+=precision
        av_recall+=recall
    
    av_precision/=5
    av_recall/=5
    f_score=(2*av_precision*av_recall)/(av_precision+av_recall)
    print('macro average')
    print('p: ',av_precision)
    print('r: ',av_recall)
    print('f: ',f_score)
    
#%%
def micro_average_score(test_topics,final_test_class):       
    total_retrieved=0
    total_relevant=0
    true_positive=0
    for each_class in test_topics:
        total_retrieved+=len(final_test_class[each_class])
        total_relevant+=len(test_topics[each_class])        
        true_positive+=len(list_intersection(test_topics[each_class],final_test_class[each_class]))
        
        
    av_precision=true_positive/total_retrieved
    av_recall=true_positive/total_relevant
    f_score=(2*av_precision*av_recall)/(av_precision+av_recall)
    print('micro average')
    print('p: ',av_precision)
    print('r: ',av_recall)
    print('f: ',f_score)

#%%
def main(test_stem,ref_topics,positional,news_body_train,all_stems_number,prob_class,test_topics,stem_doc,total_train_doc_num):
    query = input("1) Naive Bayes \n2)Naive Bayes with Mutual Information\n")
    final_test_class={}
    if int(query)==1:
        final_test_class=final_classification(query,test_stem,ref_topics,positional,news_body_train,all_stems_number,prob_class,positional)
    else:    
        mutual_features,distinct_stem=mutual_information(ref_topics,positional,stem_doc,total_train_doc_num)
        all_stems_number=len(distinct_stem)
        final_test_class=final_classification(query,test_stem,ref_topics,positional,news_body_train,all_stems_number,prob_class,mutual_features)

    
    macro_average_score(test_topics,final_test_class)
    micro_average_score(test_topics,final_test_class)
    
    
#%%
stoplist=fill_stopword(stopwords_path)
news_body_train,news_body_test,ref_topics,test_topics=read_all_documents(full_file_path,ref_topics,stoplist,test_topics)
positional,distinct_stem,stem_doc=final_preprocessing(news_body_train,ref_topics) 
test_stem=final_preprocessing_test_doc(news_body_test)


#%%
all_stems_number=len(distinct_stem)
total_train_doc_num=len(news_body_train)
prob_class=get_prob_each_class(ref_topics,total_train_doc_num)


#%%
main(test_stem,ref_topics,positional,news_body_train,all_stems_number,prob_class,test_topics,stem_doc,total_train_doc_num)
     
#%%
