import re
import json
import pandas
import random
import statistics
import nltk

nltk.download('stopwords')

from tinydb import TinyDB, Query
from nltk.corpus import stopwords

def generate_partition(total, choices):
    random.seed(1)
    ls = []
    for _ in range(total):
        selection = random.choice(choices)
        ls.append(selection)
        
    success = True
        
    s = set(choices)
    count = {}
    for i in s:
        v = ls.count(i)
        count[i] = v
        if v == 0:
            success = False
    return success, ls, count

def count_tokens(df, col):
    ls = df.loc[:, col]
    l = [len(i) for i in ls]
    average_len = statistics.mean(l)
    return average_len

# def count_tokens(df, col):
#     d
#
#     ls = df.loc[:, col]
#     l = []
#     for i in ls:
#         if i is not None:
#
#             print(i)
#             l += len(i)
#         else:
#             l += 0
#     average_len = statistics.mean(l)
#     return average_len


def db2tsv(filename, n_vocabulary):
    db = TinyDB(filename)
    data = db.all()
    
    df = pandas.DataFrame(data)
    words_document_mean = count_tokens(df, 'tokens')
    df.loc[:, 'text'] = df.loc[:, 'formated_text']
    df = df.get(['text', 'partation', 'label'])
    df = df.drop_duplicates()
    print(df.head(5))
    
    total_documents = len(df)
    generated_partition = False
    while not generated_partition:    
        partition_choices = ['train'] * 8 + ['val', 'test']
        generated_partition, partition_list, partition_count = generate_partition(total_documents, partition_choices)
    df.loc[:, 'partation'] = partition_list

    df.to_csv('data/corpus.tsv', index=False, header=False, sep='\t')
    d = {"total_documents": total_documents, "words_document_mean": words_document_mean, "vocabulary_length": n_vocabulary, "preprocessing-info": "Steps:\n  remove_punctuation\n  lemmatization\n  remove_stopwords\n  filter_words\n  remove_docs\nParameters:\n  removed words with less than 0.005 or more than 1 documents with an occurrence of the word in corpus\n  removed documents with less than 5 words", "info": {"name": "data"}, "labels": df['label'].tolist(), "total_labels": total_documents}
    
    with open('data/metadata.json', 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)
    return d



def filter_vocabulary(vocabulary):
    vocabulary_list = []
    stop_words = set(stopwords.words('english'))
    reg = r'[#@]*[a-z]+'
    for i in vocabulary:
        word = i.strip().lower()
        t = re.match(reg, word)
        if t is None:
            pass
        elif len(word) < 2:
            print('skip letter', word)
        elif word.startswith('http'):
            print('skip URL', word)
        elif word in stop_words:
            print('skip stopword', word)
        else:
            vocabulary_list.append(word)

    vocabulary_list = list(set(vocabulary_list))
    vocabulary_list.sort()
    return vocabulary_list


def process_vocabulary(filename):
    vocabulary = []
    with open(filename, 'r') as f:
        vocabulary_json = json.load(f)
        vocabulary = filter_vocabulary(vocabulary_json)
    print(len(vocabulary_json), len(vocabulary))
    
    with open('data/vocabulary.txt', 'w') as f:
        for v in vocabulary:
            f.write('{}\n'.format(v))
            
    return len(vocabulary)
        

if __name__ == '__main__':
    n_vocabulary = process_vocabulary('data/vocabulary-v1.json')
    db2tsv('data/corpus-tokenized-v1.json', n_vocabulary)
