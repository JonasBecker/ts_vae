from opts import init_opts
import numpy as np

from tensorflow.keras import Input
from tensorflow.keras.models import Model
import tensorflow as tf
import warnings
from pathlib import Path
import os
import sys
import scipy.stats

from utils import GASF # Dies muss nochmal sauber gemacht werden

def get_distribution(_min,_max,no_samples,dist_type):
    """
    Returns gaussian distribution with sigma +-3 for _min,_max bounds
    dist_type: "gaussian" / "uniform"
    return: float
    """
    if dist_type=="gaussian":
        fluegel = (_max-_min)/2
        rtn = scipy.stats.truncnorm(-3,3,loc=_min+fluegel,scale=fluegel/3).rvs(no_samples)
    elif dist_type=="uniform":
        rtn = np.random.uniform(_min,_max,no_samples)
    else:
        raise ValueError("dist_type has to be gaussian or uniform.")
    return rtn

def main():
    """
    Creates a time series dataset
    Uses opts.py for the default input options
    Saves dataset in p_out (default: data)
    """

    # 1) Input Agruments (from opts.py)
    args = init_opts()

    ## Distribution
    ds_size = args.ds_size
    dist_type = args.distribution_type
    # Time series opts
    train_split_perc = args.train_split_perc / 100
    test_split_perc = args.test_split_perc / 100
    val_split_perc = args.val_split_perc / 100
    val_split_after_test = args.val_split_after_test
    ts_len = args.ts_len
    ## Configuration der Zeitreihen:
    min_start = args.min_start
    max_start = args.max_start
    min_width = args.min_width
    max_width = args.max_width
    min_amp = args.min_amp
    max_amp = args.max_amp
    batch_size = args.batch_size
    ## Output path
    p_train = Path(args.p_out)/"train"
    p_valid = Path(args.p_out)/"valid"
    p_test = Path(args.p_out)/"test"

    
    # 2) Make Labels:
    ## 2.1) Truncated normal distribution: https://en.wikipedia.org/wiki/Truncated_normal_distribution

    # Uniform distributions:
    step_start = get_distribution(min_start,max_start,ds_size,dist_type).astype(int)
    step_width = get_distribution(min_width,max_width,ds_size,dist_type).astype(int)
    step_amp = get_distribution(min_amp,max_amp,ds_size,dist_type)

    step_end = step_start + step_width
    ## 2.2) Build labels array
    labels_raw = np.vstack([step_start, step_end, step_width, step_amp]).T
    labels = np.unique(labels_raw,axis=0)

    ## 2.3) Filter potential duplicates
    if labels.shape[0] < ds_size:
        warnings.warn(f"Warning: dataset size got smaller due equal samples. New dataset size: {len(u_indices)}")
        ds_size = len(labels)


    # 3) Make Time Series
    ts_ds = np.ones((ds_size, ts_len)) * min_amp

    for i in range(ds_size): #TODO Broadcast
        ts_ds[i, int(labels[i,0]) : int(labels[i,1] + 1)] = labels[i,3]
    print(ts_ds.shape)


    # 4) Make GAF
    inputs = Input(shape=(1,ts_len),batch_size=None,dtype=tf.float32)
    outputs = GASF()(inputs)
    gaf_model = Model(inputs = inputs, outputs = outputs, name = 'gaf_model')
    
    ts_tf_ds = tf.data.Dataset.from_tensor_slices(np.expand_dims(ts_ds,1)).batch(batch_size)

    gaf_ds = gaf_model.predict(ts_tf_ds)
    print(gaf_ds.shape)
    

    # 5) Dataset Splits
    sample_indices = np.arange(ds_size)
    np.random.shuffle(sample_indices)
    if val_split_after_test:
        val_split_perc = train_split_perc * val_split_perc
        train_split_perc = 1 - test_split_perc - val_split_perc
    
    train_indices, val_indices, test_indices = np.split(sample_indices, [int(ds_size*train_split_perc), int(ds_size*(train_split_perc + val_split_perc))])
    #gaf_ds = np.expand_dims(gaf_ds,-1) # Becker: Warum?
    
    
    # 6) Save datasets:
    for p in [p_train,p_valid,p_test]:
        if not p.exists():
            p.mkdir(parents=True)

    for ds,name in zip([ts_ds,labels,gaf_ds],["time_series","labels","gafs"]):
        np.save(p_train/name,ds[train_indices])
        np.save(p_valid/name,ds[val_indices])
        np.save(p_test/name, ds[test_indices])
    
if __name__ == '__main__':
    main()