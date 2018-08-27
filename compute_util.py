import os
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def load_data(data_path, nrows=None):
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), parse_dates=['activation_date'], nrows=nrows)
    test = pd.read_csv(os.path.join(data_path, 'test.csv'), parse_dates=['activation_date'], nrows=nrows)
    return train, test

# from https://gist.github.com/yong27/7869662
def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))
	
def parallel_df(df, func, num_cores=None, num_partitions=None):
    if num_cores is None:
        num_cores = cpu_count()
    if num_partitions is None:
        num_partitions = cpu_count()
        
    df_split = np.array_split(df, num_partitions)
    pool = Pool(processes=num_cores)
    try:
        df = pd.concat(pool.map(func, df_split))
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return df

def parallel_col_df(df, func, num_cores=None):    
    if num_cores is None:
        num_cores = cpu_count()
        
    #df = pd.DataFrame(in_df)
    m = df.shape[1]
    df_col_split = [df[:, i] for i in range(m)]
    pool = Pool(processes=num_cores)
    try:
        for i, res in enumerate(pool.map(func, df_col_split)):
            df[:, i] = res
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
        
    return df

def save_df(df, data_path, filename, compression=None):
    df.to_csv(os.path.join(data_path, '%s' % filename), index_label=False, compression=compression)

def load_df(data_path, filename):
    return pd.read_csv(os.path.join(data_path, '%s' % filename), compression='infer')
    
def rmse(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]