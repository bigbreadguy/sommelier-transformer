import os
import pyarrow.parquet as pq
import tensorflow as tf

class WineReviewDataset():
    def __init__(self, path=os.path.join('data', 'winemag-data-130k.parquet')):
        self.data = pq.read_table(path).to_pandas()
    
    def shuffle(self):
        self.data.sample(frac=1, replace=True)
    
    def transform(self, task_name='review'):
        tasks = ['review']
        if task_name not in tasks:
            raise ValueError('Invalid task name. Expected one of: %s' % tasks)
        
        switch = {
            'review': ['description', 'points', 'province', 'variety'],
        }

        return list(row.loc[switch[task_name]].to_dict() for i, row in self.data.iterrows())

def encode(example, tokenizer, encoder_max_len=54, decoder_max_len=250):
    province = example['province']
    variety = example['variety']
    points = example['points']
    review = example['description']
    
    condition_plus = f'review_me: {str(variety)} from {str(province)}, point {str(points)} </s>'
    review_plus = f'{review} </s>'
    
    encoder_inputs = tokenizer(condition_plus, truncation=True, 
                               return_tensors='tf', max_length=encoder_max_len,
                              pad_to_max_length=True)
    
    decoder_inputs = tokenizer(review_plus, truncation=True,
                               return_tensors='tf', max_length=decoder_max_len,
                              pad_to_max_length=True)
    
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, 
               'labels':target_ids, 'decoder_attention_mask':target_attention}
    return outputs

def to_tf_dataset(dataset):  
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    dataset.set_format(type='tensorflow', columns=columns)
    return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 
                    'labels':tf.int32, 'decoder_attention_mask':tf.int32,  }
    return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 
                    'labels': tf.TensorShape([None]), 'decoder_attention_mask':tf.TensorShape([None])}
    ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
    return ds

def create_dataset(dataset, cache_path=None, batch_size=4, 
                   buffer_size= 1000, shuffling=True):    
    if cache_path is not None:
        dataset = dataset.cache(cache_path)        
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=1e4):
        super().__init__()

        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        m = tf.maximum(self.warmup_steps, step)
        m = tf.cast(m, tf.float32)
        lr = tf.math.rsqrt(m)
        
        return lr 