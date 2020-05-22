import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import sys
from nltk.tokenize import word_tokenize
from os import path
import json



def term_document_matrix(data, vocab= None, document_index= 'ID', text= 'text'):
    """Calculate frequency of term in the document.
    
    parameter: 
        data: DataFrame. 
        Frequency of word calculated against the data.
        
        vocab: list of strings.
        Vocabulary of the documents    
        
        document_index: str.
        Column name for document index in DataFrame passed.
        
        text: str
        Column name containing text for all documents in DataFrame,
        
    returns:
        vocab_index: DataFrame.
        DataFrame containing term document matrix.
        """
    
    vocab_index = pd.DataFrame(columns=data[document_index], index= vocab).fillna(0)
    
    for word in vocab_index.index:
        
        for doc in data[document_index]:
            
            freq = data[data[document_index] == doc][text].values[0].count(word)
            vocab_index.loc[word,doc] = freq
    
    return vocab_index


def tf_idf_score(vocab_index, document_index, inv_df= 'inverse_document_frequency'):
    """
    Calculate tf-idf score for vocabulary in documents

    parameter:
        vocab_index: DataFrame.
        Term document matrix.
        
        document_index: list or tuple.
        Series containing document ids.
        
        inv_df: str.
        Name of the column with calculated inverse document frequencies.
        
    returns:
        vocab_index: DataFrame.
        DataFrame containing term document matrix and document frequencies, inverse document frequencies and tf-idf scores
    """
    total_docx = len(document_index)
    vocab_index['document_frequency'] = vocab_index.sum(axis= 1)
    vocab_index['inverse_document_frequency'] = np.log2( total_docx / vocab_index['document_frequency'])

    for word in vocab_index.index:
        
        for doc in document_index:
            
                tf_idf = np.log2(1 + vocab_index.loc[word,doc]) * np.log2(vocab_index.loc[word][inv_df])
                vocab_index.loc[word,'tf_idf_'+doc] = tf_idf

    return vocab_index


def query_processing(query):
    """
    Pre-processing query to accomodate calculations for tf-idf score
    
    parameter:
        query: str.
        Textual query input to the system.
        
    returns:
        query: str.
        Cleaned string.
        """
    query= re.sub('\W',' ',query)
    query= query.strip().lower()
    query= " ".join([word for word in query.split() if word not in stopwords.words('english')])
    
    return query

def query_score(vocab_index, query):
    """
    Calculate tf-idf score for query terms
    
    parameter:
        vocab_index: DataFrame.
        Term document matrix with inverse document frequency and term frequencies calculated.
        
        query: str.
        Query submitted to the system
        
    returns:
        vocab_index: DataFrame.
        Term document matrix with tf-idf scores for terms per document and query terms.
    """
    
    for word in np.unique(query.split()):
        
        freq = query.count(word)
        
        if word in vocab_index.index:
            
            tf_idf = np.log2(1+freq) * np.log2(vocab_index.loc[word].inverse_document_frequency)
            vocab_index.loc[word,"query_tf_idf"] = tf_idf
            vocab_index['query_tf_idf'].fillna(0, inplace=True)
                
    return vocab_index


def cosine_similarity(vocab_index, document_index, query_scores):
    """
    Calculates cosine similarity between the documents and query
    
    parameter:
        
        vocab_index: DataFrame.
        DataFrame containing tf-idf score per term for every document and for the query terms.
        
        document_index: list.
        List of document ids.
        
        query_scores: str.
        Column name in DataFrame containing query term tf-idf scores.
        
    returns:
        cosine_scores: Series.
        Cosine similarity scores of every document.
    """
    cosine_scores = {}
    
    query_scalar = np.sqrt(sum(vocab_index[query_scores] ** 2))
    
    for doc in document_index:
        
        doc_scalar = np.sqrt(sum(vocab_index[doc] ** 2))
        dot_prod = sum(vocab_index[doc] * vocab_index[query_scores])
        cosine = (dot_prod / (query_scalar * doc_scalar))
        
        cosine_scores[doc] = cosine
        
    return pd.Series(cosine_scores)


def retrieve_index(data,cosine_scores, document_index):
    """
    Retrieves indices for the corresponding document cosine scores
    
    parameters:
        data: DataFrame.
        DataFrame containing document ids and text.
        
        cosine_scores: Series.
        Series containing document cosine scores.
        
        document_index: str.
        Column name containing document ids in data.
        
    returns:
        data: DataFrame.
        Original DataFrame with cosine scores added as column.
    """
    
    data = data.set_index(document_index)
    data['scores'] = cosine_scores
    
    return data.reset_index().sort_values('scores',ascending=False).head(10).index



def information_system(query):
    """
    Perform a retrieval from the indexes based on the query 
    and return the document ids that are similar to the query
    
    paramters:
        query: str.
        Query submitted to the system.
        
    returns:
        indices: list.
        List of document indices which are most relevant to the query.
    """
    
    df = pd.read_csv('TagsDatabase.csv',header=None)

    df.columns = ['docID','tags']
    df.docID = pd.Series(["D"+str(ind) for ind in df.docID])

    df.tags = df.tags.str.replace(","," ")
    df.tags = df.tags.str.replace(r'\W',' ')
    df.tags = df.tags.str.strip().str.lower()
    
    if not path.exists('term_doc_matrix.csv'):    

        print("Nopeeee")
        all_text = " ".join(df.tags.values)
        vocab = np.unique(word_tokenize(all_text))
        vocab = [word for word in vocab if word not in stopwords.words('english')]

        similarity_index = term_document_matrix(df,vocab,'docID','tags')
        similarity_index = tf_idf_score(similarity_index, df.docID.values)

        similarity_index.to_csv('term_doc_matrix.csv')
        
    else:

        similarity_index = pd.read_csv('term_doc_matrix.csv')
        similarity_index = similarity_index.set_index('Unnamed: 0')
        
    query = query_processing(query)
    similarity_index = query_score(similarity_index,query)
    
    cosines = cosine_similarity(similarity_index, df.docID.values, 'query_tf_idf')
    indices = retrieve_index(df, cosines, 'docID')
    
    return json.dumps(list(indices))

if __name__ == "__main__":
    indices = information_system(sys.argv[1])
    print(indices)
