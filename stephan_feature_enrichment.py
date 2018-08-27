from os import path
import json
import pandas as pd
import multiprocessing
import numpy as np
import os
import zipfile
import gc

from utils.compute_util import load_df

def basic_enrichment(train, test, helper_data_path):
    print('Adding basic features...')
    for df in [train, test]:
        df['has_description'] = df['description'].notna()
        df['has_price'] = df['price'].notna()
        df['has_params'] = df['param_1'].notna()

        df['has_image'] = df['image'].notna()
        #df['image_top_1'].fillna(-1, inplace = True)

        #df['year'] = df['activation_date'].apply(lambda d: d.year)
        df['month'] = df['activation_date'].apply(lambda d: d.month)
        df['day'] = df['activation_date'].apply(lambda d: d.day)
        df['weekday'] = pd.to_datetime(df['activation_date']).dt.day

        #cities_geo = json.load(open(helper_data_path+'//cities_geo.json', 'r'))
        #cities_geo_df = pd.DataFrame(cities_geo).transpose().reset_index()
        #cities_geo_df.columns = ['city', 'lat', 'lng']
        #df = df.merge(cities_geo_df, on='city', how='left')

        # Add an aggregated features for ads count per user.
        user_ads = df['user_id'].value_counts()
        df['user_ads_count'] = df['user_id'].apply(lambda user_id: user_ads[user_id])

        # Merge Params to one text feature. Do not delete the params themselves.
        for col in ['param_1', 'param_2', 'param_3', 'title', 'description']:
            df[col].fillna("", inplace=True)
        df['title_description_params']= (df['title']+' '+df['description']+' '+df['param_1']+' '+df['param_2']+' '+df['param_3']).astype(str)
    
    print('Done adding basic features.')
    gc.collect()
    return train, test

def load_image_features(train, test, helper_data_path):
    print('Adding image features...')
    cols = ['img_id', 'img_size', 'img_sharpness', 'img_luminance', 'img_colorfulness', 'img_confidence', 'img_keypoints']
    img_df_train = pd.read_csv(path.join(helper_data_path, 'train_img_features.zip'), compression='infer', usecols=cols)
    img_df_test = pd.read_csv(path.join(helper_data_path, 'test_img_features.zip'), compression='infer', usecols=cols)
    img_df_train = img_df_train.rename(index=str, columns={'img_id': 'image'})
    img_df_test = img_df_test.rename(index=str, columns={'img_id': 'image'})
    train = train.merge(img_df_train, on='image', how='left')
    test = test.merge(img_df_test, on='image', how='left')
    
    for df in [train, test]:
        # Log scale skewed features (exploration done in the relevant notebook).
        log_cont_ord_features = ['img_sharpness', 'img_keypoints']
        for col in log_cont_ord_features:
            df['log_'+col] = np.log1p(df[col])
            df['log_'+col] = np.log1p(df[col])

        # Clip outliers.
        columns = ['img_sharpness', 'img_colorfulness', 'img_keypoints', 'log_img_sharpness', 'log_img_keypoints']
        for col in columns:
            df[df[col] >= df[col].quantile(0.99)][col] = df[col].quantile(0.99)
            df[df[col] >= df[col].quantile(0.99)][col] = df[col].quantile(0.99)
    
    print('Done loading image features.')
    gc.collect()
    return train, test

def load_text_features(train, test, helper_data_path):
    print('Loading text features...')
    
    # Stephan's nlp
    def get_df(filename):
        final_nlp_df = None
        with zipfile.ZipFile(path.join(helper_data_path, filename), 'r') as zip_ref:
            for name in zip_ref.namelist():
                nlp_df = pd.read_pickle(zip_ref.open(name))
                nlp_df = nlp_df[['item_id', 'title_word_count',
                                 'description_non_regular_chars_ratio',
                                 'description_word_count',
                                'merged_params_word_count',
                                'description_sentence_count', 
                                'description_words/sentence_ratio',
                                'title_capital_letters_ratio',
                                'description_capital_letters_ratio',
                                'title_non_regular_chars_ratio',
                                'title_num_of_newrow_char',
                                'description_num_of_newrow_char',
                                'title_num_adj',
                                'title_num_nouns',
                                'title_adj_to_len_ratio',
                                'title_noun_to_len_ratio',
                                'description_num_adj',
                                'description_num_nouns',
                                'description_adj_to_len_ratio',
                                'description_noun_to_len_ratio',
                                'title_first_noun_stemmed',                                
                                'title_second_noun_stemmed',
                                'title_third_noun_stemmed',
                                'description_first_noun_stemmed',
                                'description_second_noun_stemmed',
                                'description_third_noun_stemmed',
                                'title_first_adj_stemmed',
                                'title_second_adj_stemmed',
                                'title_third_adj_stemmed',
                                'description_first_adj_stemmed',
                                'description_second_adj_stemmed',
                                'description_third_adj_stemmed',
                                'title_sentiment',
                                'description_sentiment'
                ]]
                if final_nlp_df is not None:
                    final_nlp_df = pd.concat([final_nlp_df, nlp_df])
                else:
                    final_nlp_df = nlp_df
        return final_nlp_df
        
    train = train.merge(get_df('train_NLP_enriched.zip'), on='item_id', how='left')
    if test is not None:
        test = test.merge(get_df('test_NLP_enriched.zip'), on='item_id', how='left')
    
    # tf-idf
    # print('loading tfidf features...')
    # tfidf_df = load_df(helper_data_path, 'train_tfidf_svd.csv.gz')
    # # train = train.merge(tfidf_df, on='item_id', how='left')
    # train = pd.concat([train,tfidf_df],axis=1)
    # if test is not None:
    #    tfidf_df = load_df(helper_data_path, 'test_tfidf_svd.csv.gz')
    #    #test = pd.concat([test,tfidf_df])
    #    test = pd.concat([test,tfidf_df],axis=1)    
        
    print('Done loading text features.')
    gc.collect()
    return train, test

def add_aggregated_features(train, test, helper_data_path):
    aggregated_features = load_df(helper_data_path, 'aggregated_features.csv.gz')
    train = add_aggregated_features_inner(train, aggregated_features)
    test = add_aggregated_features_inner(test, aggregated_features)
    return train, test

def add_aggregated_features_inner(df, aggregated_features):
    print('Loading aggregated features...')
    
    # Load downloaded aggregated features.
    # https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm/notebook
    df = df.merge(aggregated_features, on='user_id', how='left')
    df['avg_days_up_user'].fillna(0, inplace=True)
    df['avg_times_up_user'].fillna(0, inplace=True)
    df['n_user_items'].fillna(0, inplace=True)
    
    print('Done loading aggregated features.')
    gc.collect()
    return df

def numeric_features_cleaning(train, test, helper_data_path):
    print('Cleaning and completing numeric features...')
    
    for df in [train, test]:
        # Log scale skewed features.
        log_cont_ord_features = ['item_seq_number', 'price', 'description_word_count']
        for col in log_cont_ord_features:
            df['log_'+col] = np.log1p(df[col])
            df['log_'+col] = np.log1p(df[col])

        # Clip outliers.
        columns = ['price', 'description_word_count', 'log_price', 'log_description_word_count']
        for col in columns:
            df[df[col] >= df[col].quantile(0.99)][col] = df[col].quantile(0.99)
            df[df[col] >= df[col].quantile(0.99)][col] = df[col].quantile(0.99)

    print('Done cleaning numeric features.')
    gc.collect()
    return train, test

def complete_price(train, test, helper_data_path):
    print('Completing price...')
    
    price_df_train_class = pd.read_csv(path.join(helper_data_path, 'completed_train_price.csv.gz'), compression='infer')
    price_df_test_class = pd.read_csv(path.join(helper_data_path, 'completed_test_price.csv.gz'), compression='infer')
    train = train.merge(price_df_train_class, on='item_id', how='left')
    test = test.merge(price_df_test_class, on='item_id', how='left')
    
    print('Done loading log_price_regression.')
    del price_df_train_class, price_df_test_class
    gc.collect()
    return train, test
    
def complete_image_top_1(train, test, helper_data_path):
    print('Completing image_top_1 features...')
    
    # Completion by classification.
    img_df_train_class = pd.read_csv(path.join(helper_data_path, 'completed_train_image_top_1_class.csv.gz'), compression='infer')
    img_df_test_class = pd.read_csv(path.join(helper_data_path, 'completed_test_image_top_1_class.csv.gz'), compression='infer')
    train = train.merge(img_df_train_class, on='item_id', how='left')
    test = test.merge(img_df_test_class, on='item_id', how='left')
    
    # Completion by regression.
    img_df_train_reg = pd.read_csv(path.join(helper_data_path, 'completed_train_image_top_1_reg.csv.gz'), compression='infer')
    img_df_test_reg = pd.read_csv(path.join(helper_data_path, 'completed_test_image_top_1_reg.csv.gz'), compression='infer')
    train = train.merge(img_df_train_reg, on='item_id', how='left')
    test = test.merge(img_df_test_reg, on='item_id', how='left')
    
    print('Done loading image_top_1 completions.')
    del img_df_train_class, img_df_test_class, img_df_train_reg, img_df_test_reg
    gc.collect()
    return train, test

