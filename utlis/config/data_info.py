import tensorflow as tf

casia_B_train = {
    "feature": {
        "angle": tf.foat32,
        "subject": tf.float32,
        "data_row": tf.string
    },

    "resolution": {
        "height": 64,
        "width": 64,
        "channel": None
    },

    "training_info": {
        "size": 
        "batch_size": 64,
        "shuffle": True
    }

}