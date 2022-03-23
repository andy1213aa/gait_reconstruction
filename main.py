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

    encoder = Encoder().model((64, 64), 5)
    view_transform_layer = View_Transform_Layer().model(128)
    generator = Generator(128, 11, (64, 64)).model()
    discriminator = Discriminator(62, 5, 11)
    view_angle_classfier = View_Angle_Classfier(11).model(128)


    d_optimizer = tf.keras.optimizers.RMSprop(lr=2e-4, decay=1e-4)
    g_optimizer = tf.keras.optimizers.RMSprop(lr=5e-5, decay=1e-4) #also used in "encoder", "view_transform_layer" and "view_angle_classfier"

    while iteration < 


if __name__ == '__main__':
    main()
