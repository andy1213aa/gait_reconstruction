import tensorflow.math


def Pixelwise_Loss(fake, real):
    L1norm = tensorflow.math.reduce_mean(tensorflow.math.abs(fake-real))
    return L1norm


def Generator_Loss(fake_logit):
    g_loss = -tensorflow.reduce_mean(tensorflow.math.log(fake_logit))
    return g_loss

def Discriminator_Loss(real_logit, fake_logit):
    real_loss = -tensorflow.reduce_mean(tensorflow.math.log(real_logit))
    fake_loss = -tensorflow.reduce_mean(tensorflow.math.log(1-fake_logit))
    return real_loss, fake_loss

