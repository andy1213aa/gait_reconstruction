'''
Implement "Multi-Task GANs for View-Specific Feature Learning in Gait Recognition"

'''
from dis import dis
import tensorflow as tf
from model.encoder import Encoder
from model.generator import Generator
from model.discriminator import Discriminator
from model.view_angle_classfier import View_Angle_Classfier
from model.view_transform_layer import View_Transform_Layer
from utlis import view_transform_encoder
from utlis.loss_function import Generator_Loss   
from utlis.loss_function import Discriminator_Loss


def main():

    @tf.function
    def train_generator(real_data):
        with tf.GradientTape(persistent=True) as tape:
            view_specific_feature = encoder(real_data, training = True) 
            view_transform_vector = view_transform_layer(view_specific_feature, real_data['angle'])
            concate_feature = tf.concate(view_specific_feature, view_transform_vector)
            predict_silhouette = generator(concate_feature, training = True)
            gFake_logit = discriminator([real_data['image_raw'], predict_silhouette],training = True)
            gFake_loss = Generator_Loss(gFake_logit)
            
            disparate = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.abs(predict_silhouette-real_data[1]), axis=[1, 2])))
            gLoss = gFake_loss + disparate

        encoder_gradients = tape.gradient(gLoss, encoder_optimizer.trainable_variables)
        view_transform_layer_gradients = tape.gradient(gLoss, view_transform_layer_optimizer.trainable_variables)
        generator_gradients = tape.gradient(gLoss, generator_optimizer.trainable_variables)
        

        encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
        view_transform_layer_optimizer.apply_gradients(zip(view_transform_layer_gradients, view_transform_layer.trainable_variables))
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        return gFake_loss

    @tf.function
    def train_discriminator(real_data):
     
        with tf.GradientTape() as t:
            
            view_specific_feature = encoder(real_data, training = True) 
            view_transform_vector = view_transform_layer(view_specific_feature, real_data['angle'])
            concate_feature = tf.concate(view_specific_feature, view_transform_vector)
            predict_silhouette = generator(concate_feature, training = True)
            
            real_logit = discriminator(real_data['iage_raw'] ,training = True)
            fake_logit = discriminator(predict_silhouette, training = True)
            
            real_loss, fake_loss = Discriminator_Loss(real_logit, fake_logit)
            dLoss = (fake_loss + real_loss)

        D_grad = t.gradient(dLoss, dis.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(D_grad, dis.trainable_variables))
        return real_loss, fake_loss


    encoder = Encoder().model((64, 64, 1), 5)
    view_transform_layer = View_Transform_Layer().model(128)
    generator = Generator(128, 11, (64, 64)).model()
    discriminator = Discriminator(62, 5, 11)
    view_angle_classfier = View_Angle_Classfier(11).model(128)


    discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=2e-4, decay=1e-4)
    generator_optimizer = tf.keras.optimizers.RMSprop(lr=5e-5, decay=1e-4) 
    encoder_optimizer = tf.keras.optimizers.RMSprop(lr=5e-5, decay=1e-4) 
    view_transform_layer_optimizer  = tf.keras.optimizers.RMSprop(lr=5e-5, decay=1e-4) 

    #also used in "encoder", "view_transform_layer" and "view_angle_classfier"

    iteration = 0
    while iteration < 10000:
        for step, real_data in enumerate(training_batch):
            gen_fake_loss = train_generator(real_data)
            dis_real_loss = train_discriminator(real_data)
        
        
        iteration += 1


if __name__ == '__main__':
    main()
