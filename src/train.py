import os
import datetime

import numpy as np
import tensorflow as tf

from src.model import tokenizer, WineReviewT5
from src.dataset import WineReviewDataset, encode, to_tf_dataset, create_dataset

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

def train():
    train_test_split_ratio = 0.2
    warmup_steps = 1e4
    batch_size = 4
    encoder_max_len = 250
    decoder_max_len = 54
    buffer_size = 1000

    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    log_dir = os.path.join(data_dir, 'experiments', 't5', 'logs')
    save_path = os.path.join(data_dir, 'experiments', 't5', 'models')
    cache_path_train = os.path.join(data_dir, 'cache', 't5.train')
    cache_path_test = os.path.join(data_dir, 'cache', 't5.test')

    dataset = WineReviewDataset()
    dataset.shuffle()
    dataset = dataset.transform('review')
    
    cut = int(len(dataset) * train_test_split_ratio)
    train_dataset, valid_dataset = dataset[:cut], dataset[cut:]

    ntrain = len(train_dataset)
    nvalid = len(valid_dataset)
    steps = int(np.ceil(ntrain/batch_size))
    valid_steps = int(np.ceil(nvalid/batch_size))

    tokenizer = tokenizer()
    train_ds = train_dataset.map(lambda x: encode(x, tokenizer=tokenizer))
    valid_ds = valid_dataset.map(lambda x: encode(x, tokenizer=tokenizer))

    tf_train_ds = to_tf_dataset(train_ds)
    tf_valid_ds = to_tf_dataset(valid_ds)

    tf_train_ds = create_dataset(tf_train_ds, batch_size=batch_size,
                        shuffling=True, cache_path = None)
    tf_valid_ds = create_dataset(tf_valid_ds, batch_size=batch_size,
                        shuffling=False, cache_path = None)

    start_profile_batch = steps + 10
    stop_profile_batch = start_profile_batch + 100
    profile_range = f'{start_profile_batch},{stop_profile_batch}'

    log_path = os.path.join(log_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                    log_dir=log_path, histogram_freq=1,
                                    update_freq=20,profile_batch=profile_range)

    checkpoint_filepath = os.path.join(save_path, 'T5-{epoch:04d}-{val_loss:.4f}.ckpt')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=checkpoint_filepath,
                                    save_weights_only=False,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)

    callbacks = [tensorboard_callback, model_checkpoint_callback] 
    metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')]

    learning_rate = CustomSchedule()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = WineReviewT5.from_pretrained("t5-base")
    model.compile(optimizer=optimizer, metrics=metrics)

    epochs_done = 0
    model.fit(tf_train_ds, epochs=5, steps_per_epoch=steps, callbacks=callbacks, 
               validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)
    model.save_pretrained(save_path)