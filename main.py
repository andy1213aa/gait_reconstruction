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
from utlis import create_training_data
from utlis.save_model import Save_Model
from utlis.config.data_info import OU_MVLP_train


def main():

    @tf.function
    def train_generator(subject, angle, image):

        with tf.GradientTape(persistent=True) as tape:

            view_specific_feature = encoder(image[0], training=True)

            view_transform_vector = view_transform_layer(
                view_specific_feature, angle)
            concate_feature = tf.concate(
                view_specific_feature, view_transform_vector)
            predict_silhouette = generator(concate_feature, training=True)
            fake_logit = discriminator(
                [image[1], predict_silhouette], training=True)
            fake_loss = Generator_Loss(fake_logit)

            disparate = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(
                tf.math.abs(predict_silhouette-image[1]), axis=[1, 2, 3])))
            gen_total_loss = fake_loss + disparate

        encoder_gradients = tape.gradient(
            gen_total_loss, encoder_optimizer.trainable_variables)

        view_transform_layer_gradients = tape.gradient(
            gen_total_loss, view_transform_layer_optimizer.trainable_variables)

        generator_gradients = tape.gradient(
            gen_total_loss, generator_optimizer.trainable_variables)

        encoder_optimizer.apply_gradients(
            zip(encoder_gradients, encoder.trainable_variables))

        view_transform_layer_optimizer.apply_gradients(
            zip(view_transform_layer_gradients, view_transform_layer.trainable_variables))

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables))

        return gen_total_loss, disparate

    @tf.function
    def train_discriminator(subject, angle, image):

        with tf.GradientTape() as t:

            view_specific_feature = encoder(image[0], training=True)

            view_transform_vector = view_transform_layer(
                view_specific_feature, angle)
            concate_feature = tf.concate(
                view_specific_feature, view_transform_vector)
            predict_silhouette = generator(concate_feature, training=True)

            fake_logit = discriminator(
                [image[1], predict_silhouette], training=True)

            real_logit = discriminator(image[1], training=True)

            real_loss, fake_loss = Discriminator_Loss(real_logit, fake_logit)
            dLoss = (fake_loss + real_loss)

        D_grad = t.gradient(dLoss, dis.trainable_variables)
        discriminator_optimizer.apply_gradients(
            zip(D_grad, dis.trainable_variables))
        return real_loss, fake_loss
        
    angle_num = OU_MVLP_train['resolution']['angle_nums']

    encoder = Encoder().model((64, 64))
    view_transform_layer = View_Transform_Layer(128).model(128, angle_num)
    generator = Generator(128, angle_num, (64, 64)).model()
    discriminator = Discriminator().model()
    view_angle_classfier = View_Angle_Classfier(angle_num).model(128)

    discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=2e-4, decay=1e-4)
    generator_optimizer = tf.keras.optimizers.RMSprop(lr=5e-5, decay=1e-4)
    encoder_optimizer = tf.keras.optimizers.RMSprop(lr=5e-5, decay=1e-4)
    view_transform_layer_optimizer = tf.keras.optimizers.RMSprop(
        lr=5e-5, decay=1e-4)

    training_batch = create_training_data('OU_MVLP')
    save_model = Save_Model(generator, discriminator, )

    iteration = 0
    while iteration < 10000:
        for step, batch in enumerate(training_batch):

            batch_subjects, batch_angles, batch_images = batch

            gen_fake_loss, disparate = train_generator(
                batch_subjects, batch_angles, batch_images)

            dis_real_loss, dis_fake_loss = train_discriminator(
                batch_subjects, batch_angles, batch_images)

        iteration += 1
        print(f'Epoch: {iteration:6} Batch: {step:3} Disparate:{disparate:4.5} G_loss: {gen_fake_loss:4.5} D_real_loss: {dis_real_loss:4.5} D_fake_loss: {dis_fake_loss:4.5}')

        if iteration % 1 == 0:
            save_model.save()


if __name__ == '__main__':
    main()
