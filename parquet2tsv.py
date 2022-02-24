#!/usr/bin/env python3

#test111111111111111111111123

import json
import pandas
import pandas as pd
import random
import statistics
import nltk
import datetime
#import tinydb # TinyDB, Query
import logging
import argparse

#try:
#    from nltk.corpus import stopwords
#except:
#    nltk.download('stopwords')

from nltk.corpus import stopwords
from process2tsv import generate_partition, count_tokens

print('start2')

parser = argparse.ArgumentParser()
parser.add_argument('--logpath', type=str, default='log-parquet2tsv.log')
parser.add_argument('--parquetfile', type=str, default='data/processed_and_fixed_corpus_parquet_data.parquet')
parser.add_argument('--textcol', type=str, default='text')
parser.add_argument('--partitioncol', type=str, default='partition')
parser.add_argument('--tokencol', type=str, default='tokens')
parser.add_argument('--labelcol', type=str, default='label')
parser.add_argument('--vocabularyfile', type=str, default='data/vocabulary.json')
parser.add_argument('--corpusfile', type=str, default='data/corpus.tsv')
parser.add_argument('--metafile', type=str, default='data/metadata.json')


def log_str(*ls):
    line = [str(i) for i in ls]
    line = ' '.join(line)
    logging.info(line)
    print(line)
    
    
def expand_table(df, labelcol):
    filtered_index = []
    user_id = []
    tweet_id = []
    time = []
    label = []
    
    year = []
    month = []
    day = []
    
    with open('invalid-time.txt', 'w') as f:
        f.write('')
    
    for i in df.index:
        label_str = df.loc[i, labelcol]
        if label_str is not None:
            ls = label_str.split('.')
            if len(ls) == 5:
                _, uid, tid, timestamp, l = ls
                if ':' in timestamp:
                    dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    filtered_index.append(i)
                    
                    user_id.append(uid)
                    tweet_id.append(tid)
                    time.append(dt)
                    label.append(l)

                    year.append(dt.year)
                    month.append(dt.month)
                    day.append(dt.day)
                else:
                    with open('invalid-time.txt', 'a') as f:
                        f.write(label_str + '\n')
            else:
                with open('invalid-time.txt', 'a') as f:
                    f.write('invalid line: ' + label_str +'\n')
    
    df = df.loc[filtered_index, :]
    df['user_id'] = user_id
    df['tweet_id'] = tweet_id
    df['time'] = time
    df['label'] = label
    df['year'] = year
    df['month'] = month
    df['day'] = day
    return df


def get_label(labels):
    c = list(set(labels))
    c.sort()
    count = 0
    l = None
    for i in c:
        cc = labels.count(i)
        if cc > count:
            l = i
            count = cc
            
    return l, count


def split_table(df, textcol, tokencol, labelcol):
    db = tinydb.TinyDB('data/daily-tweeted.json')
    
    start_day = min(df['time'])
    end_day = max(df['time'])
    users = pd.unique(df['user_id'])
    users.sort()
    current_day = start_day
    with open('invalid-tokens.txt', 'w') as f:
        f.write('')
   
    while current_day <= end_day:
        for uid in users:
            df_day = df.loc[ (df['user_id'] == uid) & (df['year'] == current_day.year) & (df['month'] == current_day.month) & (df['day'] == current_day.day) ]
            if len(df_day) == 0:
                with open('invalid-tokens.txt', 'a') as f:
                    f.write('{}-{}: {}.\n'.format(uid, current_day.date(), df_day.shape))
            else:
                text = ' '.join(df_day[textcol].tolist())
                tokens = set([])
                for tk in df_day[tokencol]:
                    tokens.update(set(tk))
                tokens = list(tokens)
                if len(tokens) == 0:
                    with open('invalid-tokens.txt', 'a') as f:
                        f.write('invalid tokens: ' + str(df_day[tokencol].tolist()) +'\n')

                labels = df_day[labelcol].tolist()
                class_label, lcount = get_label(labels)
                if lcount < len(labels):
                    log_str(uid, class_label, lcount, labels)

                item_label = '{}.{}.{}'.format(uid,current_day.date(),class_label)
                item = {'user_id':uid, 'date':str(current_day.date()), 'year':current_day.year, 'month':current_day.month, 'day':current_day.day, textcol:text, tokencol:tokens, labelcol:item_label}
                with open('invalid-tokens.txt', 'a') as f:
                    f.write('day tweeted item: ' + str(item) +'\n')
                db.insert(item)
            
        current_day = current_day + datetime.timedelta(days=1)
        
    return db, start_day, end_day

        
        
def process_table(args, dftrain, dfval, dftest):
    # merge csv
    n_train = len(dftrain)
    n_val = len(dfval)
    n_test = len(dftest)
    dftrain[args.partitioncol] = ['train'] * n_train
    dfval[args.partitioncol] = ['val'] * n_val
    dftest[args.partitioncol] = ['test'] * n_test
    
    df = pd.concat([dftrain, dfval, dftest], sort=False)
    
    words_document_mean = 0
    try:
        words_document_mean = count_tokens(df, args.tokencol)
    except:
        print("words_document_mean is set as 0")
    log_str('words_document_mean', words_document_mean)

    # get vocabulary list
    vocabulary = set([])
    for i in df[args.tokencol]:
        vocabulary.update(set(i))
    vocabulary = list(vocabulary)
    vocabulary.sort()
    with open(args.vocabularyfile, 'w') as f:
        json.dump(vocabulary, f, sort_keys=True, indent=4)
    n_vocabulary = len(vocabulary)
    log_str('n_vocabulary', n_vocabulary)
    with open(args.vocabularyfile.replace('.json', '.txt'), 'w') as f:
        for v in vocabulary:
            f.write(v)
            f.write('\n')   
    log_str('finish writing vocabulary file')
    
    # write tsv file
    df = df.get([args.textcol, args.partitioncol, args.labelcol])
    df = df.drop_duplicates()
    total_documents = len(df)
    log_str('write to tsv')
    df.to_csv(args.corpusfile, index=False, header=False, sep='\t')
    
    # write meta data
    log_str('write meta data')
    d = {
        "total_documents": total_documents, 
        "words_document_mean": words_document_mean, 
        "vocabulary_length": n_vocabulary, 
        "last-training-doc": n_train, 
        "last-validation-doc": n_val,
        "preprocessing-info": "Steps:\n  remove_punctuation\n  lemmatization\n  remove_stopwords\n  filter_words\n  remove_docs\nParameters:\n  removed words with less than 0.005 or more than 1 documents with an occurrence of the word in corpus\n  removed documents with less than 5 words", 
        "info": {"name": "data"}, 
        "labels": df['label'].tolist(), 
        "total_labels": total_documents
    }
    
    with open(args.metafile, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)


def main(args):
    print('reading parquet', args.parquetfile)
    df = pandas.read_parquet(args.parquetfile)
    log_str(df.head(5))
    log_str(list(df))
    
    df = expand_table(df, args.labelcol)
    db, start_day, end_day = split_table(df, args.textcol, args.tokencol, args.labelcol)

    log_str("start generating tsv and meta data")
    
    dfval = df.loc[0:100, [args.textcol, args.labelcol, args.tokencol]]
    dftest = df.loc[100:200, [args.textcol, args.labelcol, args.tokencol]]
    
    df_daily = pandas.DataFrame(db.all()).loc[:, [args.textcol, args.labelcol, args.tokencol]]

    log_str("wrote to tsv")
    log_str("wrote to tsv")
    process_table(args, df_daily, dfval, dftest)
    

    
    
if __name__ == '__main__':
    print('begain')
    args = parser.parse_args()
    
    logging.basicConfig(
        filename=args.logpath,
        filemode='a',
        format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
    )
    
    main(args)

