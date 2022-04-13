import tensorflow as tf

training_info = {
    'save_model': {
        'logdir': r'C:\Users\User\Desktop\Aaron\College-level Applied Research\log\gait_training_log'
    }
}


casia_B_train = {
    "feature": {
        "angle": tf.float32,
        "subject": tf.float32,
        "data_row": tf.string
    },

    "resolution": {
        "height": 64,
        "width": 64,
        "channel": None
    },

    "training_info": {
        "size": 73,
        "batch_size": 64,
        "shuffle": True
    }

}

OU_MVLP_train = {

    "feature": {
        "subject": tf.float32,
        "angles":  tf.string,
        "images": tf.string
    },

    "resolution": {
        "height": 64,
        "width": 64,
        "channel": 1,
        "angle_nums": 14
    },

    "training_info": {
        "tfrecord_path": r'C:\Users\User\Desktop\Aaron\College-level Applied Research\gait_recognition\OU_MVLP_GEI_TFRECORD\GEI_00_float32.tfrecords',
        "data_num": 10300,
        "batch_size": 64,
        "shuffle": True
    }
}
