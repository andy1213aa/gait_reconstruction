'''
Implement "Multi-Task GANs for View-Specific Feature Learning in Gait Recognition"

'''
import tensorflow as tf
from model.encoder import Encoder
from model.generator import Generator
from model.discriminator import Discriminator
from model.view_angle_classfier import View_Angle_Classfier
from model.view_transform_layer import View_Transform_Layer


def main():

    @tf.function
    def train_generator(real_data):
        with tf.GradientTape() as tape:
            view_specific_feature = encoder(real_data, training = True) 
            view_transform_layer = 

            gFake_logit = dis([real_data[0], fake_data_by_random_parameter],training = True)
            gFake_loss = generator_loss(gFake_logit)
            disparate = 5e-1*tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.abs(fake_data_by_random_parameter-real_data[1])**2, axis=[1, 2, 3, 4])))
            gLoss = gFake_loss + disparate
        gradients = tape.gradient(gLoss, gen.trainable_variables)
        genOptimizer.apply_gradients(zip(gradients, gen.trainable_variables))
        return gFake_loss, gFake_logit, disparate
    
    @tf.function
    def train_discriminator(real_data):
     
        with tf.GradientTape() as t:

            fake_data = gen(real_data[0],training = True)
            real_logit = dis([real_data[0], real_data[1]] ,training = True)
            fake_logit = dis([real_data[0], fake_data],training = True)
            real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)
            dLoss = (fake_loss + real_loss)

        D_grad = t.gradient(dLoss, dis.trainable_variables)
        disOptimizer.apply_gradients(zip(D_grad, dis.trainable_variables))
        return real_loss ,  fake_loss,  real_logit,fake_logit


    encoder = Encoder().model((64, 64, 1), 5)
    view_transform_layer = View_Transform_Layer().model(128)
    generator = Generator(128, 11, (64, 64)).model()
    discriminator = Discriminator(62, 5, 11)
    view_angle_classfier = View_Angle_Classfier(11).model(128)


    d_optimizer = tf.keras.optimizers.RMSprop(lr=2e-4, decay=1e-4)
    g_optimizer = tf.keras.optimizers.RMSprop(lr=5e-5, decay=1e-4) #also used in "encoder", "view_transform_layer" and "view_angle_classfier"

    iteration = 0
    while iteration < 10000:
        for step, real_data in enumerate(training_batch):
            train_generator(real_data)
        iteration += 1


if __name__ == '__main__':
    main()
