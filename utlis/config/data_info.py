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
        "channel": None
    },

    "training_info": {
        "tfrecord_path": r'F:\OU_MVLP\OU_MVLP_GEI_TFRECORD\GEI_00_float32.tfrecords',
        "data_num": 10300,
        "batch_size": 64,
        "shuffle": True
    }
}
