import tensorflow as tf


def Pixelwise_Loss(fake, real):
    L1norm = tf.reduce_mean(tf.math.abs(fake-real))
    return L1norm


def Generator_Loss(fake_logit):

    # if type == 'WGAN' or 'hinge':
    g_loss = -tf.reduce_mean(fake_logit)

    # #type == 'origin':
    # g_loss = -tf.reduce_mean(tf.math.log(gfake_logit))

    return g_loss


def Discriminator_Loss(real_logit, fake_logit):
    # # if type == 'WGAN':
    # real_loss = -tf.reduce_mean(real_logit)
    # fake_loss = tf.reduce_mean(fake_logit)

    # elif type == 'hinge':
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))

    # # type == 'origin':
    # real_loss = -tf.reduce_mean(tf.math.log(real_logit))
    # fake_loss = -tf.reduce_mean(tf.math.log(1-fake_logit))
    return real_loss, fake_loss
