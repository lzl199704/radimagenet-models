"""
Script to transform the weights of the ResNetmodel from Keras to Pytorch.
Script based on https://github.com/BMEII-AI/RadImageNet/issues/3#issuecomment-1232417600
and https://discuss.pytorch.org/t/transferring-weights-from-keras-to-pytorch/9889
"""
import argparse
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, DenseNet121
import torch
from models.inception_v3 import Inception3
#from torchvision.models import resnet50, densenet121,inception_v3

os.environ["CUDA_VISIBLE_DEVICES"]='2'

torch.set_printoptions(precision=10)

input_path = "/raid/data/yanglab/RadImageNet_models/RadImageNet-InceptionV3-224-512-081421_notop.h5"
out_path = "RadImageNet-InceptionV3_notop.pth"

def convert_conv(pytorch_conv, tf_conv):
    pytorch_conv.weight.data = torch.tensor(np.transpose(tf_conv.kernel.numpy(), (3, 2, 0, 1)))
    #pytorch_conv.bias.data = torch.tensor(tf_conv.bias.numpy())
    return pytorch_conv


def convert_bn(pytorch_bn, tf_bn):
    #pytorch_bn.weight.data = torch.tensor(tf_bn.gamma.numpy())
    pytorch_bn.bias.data = torch.tensor(tf_bn.beta.numpy())
    pytorch_bn.running_mean.data = torch.tensor(tf_bn.moving_mean.numpy())
    pytorch_bn.running_var.data = torch.tensor(tf_bn.moving_variance.numpy())
    return pytorch_bn

def convert_mix_5(pytorch_mix, keras_model,mix_name):
    if mix_name=='5b':
        keras_layer_list= [5, 6,7,8,9,10,11] ##[64,48,64,64,96,96,32]
    elif mix_name == '5c':
        keras_layer_list = list(range(12,19,1))
    elif mix_name =='5d':
        keras_layer_list = list(range(19,26,1))
    
    pytorch_mix.branch1x1.conv = convert_conv(pytorch_mix.branch1x1.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[0])))
    pytorch_mix.branch1x1.bn = convert_bn(pytorch_mix.branch1x1.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[0])))    
    pytorch_mix.branch5x5_1.conv = convert_conv(pytorch_mix.branch5x5_1.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[1])))
    pytorch_mix.branch5x5_1.bn = convert_bn(pytorch_mix.branch5x5_1.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[1])))
    pytorch_mix.branch5x5_2.conv = convert_conv(pytorch_mix.branch5x5_2.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[2])))
    pytorch_mix.branch5x5_2.bn = convert_bn(pytorch_mix.branch5x5_2.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[2])))
    pytorch_mix.branch3x3dbl_1.conv = convert_conv(pytorch_mix.branch3x3dbl_1.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[3])))
    pytorch_mix.branch3x3dbl_1.bn = convert_bn(pytorch_mix.branch3x3dbl_1.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[3])))
    pytorch_mix.branch3x3dbl_2.conv = convert_conv(pytorch_mix.branch3x3dbl_2.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[4])))
    pytorch_mix.branch3x3dbl_2.bn = convert_bn(pytorch_mix.branch3x3dbl_2.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[4])))
    pytorch_mix.branch3x3dbl_3.conv = convert_conv(pytorch_mix.branch3x3dbl_3.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[5])))
    pytorch_mix.branch3x3dbl_3.bn = convert_bn(pytorch_mix.branch3x3dbl_3.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[5])))
    pytorch_mix.branch_pool.conv = convert_conv(pytorch_mix.branch_pool.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[6])))
    pytorch_mix.branch_pool.bn = convert_bn(pytorch_mix.branch_pool.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[6])))
    return pytorch_mix

def convert_mix_6(pytorch_mix, keras_model,mix_name):
    if mix_name=='6a':
        keras_layer_list= [26,27,28,29] ##[64,48,64,64,96,96,32]
        pytorch_mix.branch3x3.conv = convert_conv(pytorch_mix.branch3x3.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[0])))
        pytorch_mix.branch3x3.bn = convert_bn(pytorch_mix.branch3x3.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[0])))    
        pytorch_mix.branch3x3dbl_1.conv = convert_conv(pytorch_mix.branch3x3dbl_1.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[1])))
        pytorch_mix.branch3x3dbl_1.bn = convert_bn(pytorch_mix.branch3x3dbl_1.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[1])))
        pytorch_mix.branch3x3dbl_2.conv = convert_conv(pytorch_mix.branch3x3dbl_2.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[2])))
        pytorch_mix.branch3x3dbl_2.bn = convert_bn(pytorch_mix.branch3x3dbl_2.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[2])))
        pytorch_mix.branch3x3dbl_3.conv = convert_conv(pytorch_mix.branch3x3dbl_3.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[3])))
        pytorch_mix.branch3x3dbl_3.bn = convert_bn(pytorch_mix.branch3x3dbl_3.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[3])))
        return pytorch_mix
    elif mix_name == '6b':
        keras_layer_list = list(range(30,40,1))
    elif mix_name =='6c':
        keras_layer_list = list(range(40,50,1))
    elif mix_name == '6d':
        keras_layer_list = list(range(50,60,1))
    elif mix_name == '6e':
        keras_layer_list = list(range(60,70,1))
    pytorch_mix.branch1x1.conv = convert_conv(pytorch_mix.branch1x1.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[0])))
    pytorch_mix.branch1x1.bn = convert_bn(pytorch_mix.branch1x1.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[0])))
    pytorch_mix.branch7x7_1.conv = convert_conv(pytorch_mix.branch7x7_1.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[1])))
    pytorch_mix.branch7x7_1.bn = convert_bn(pytorch_mix.branch7x7_1.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[1])))
    pytorch_mix.branch7x7_2.conv = convert_conv(pytorch_mix.branch7x7_2.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[2])))
    pytorch_mix.branch7x7_2.bn = convert_bn(pytorch_mix.branch7x7_2.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[2])))
    pytorch_mix.branch7x7_3.conv = convert_conv(pytorch_mix.branch7x7_3.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[3])))
    pytorch_mix.branch7x7_3.bn = convert_bn(pytorch_mix.branch7x7_3.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[3])))
    pytorch_mix.branch7x7dbl_1.conv = convert_conv(pytorch_mix.branch7x7dbl_1.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[4])))
    pytorch_mix.branch7x7dbl_1.bn = convert_bn(pytorch_mix.branch7x7dbl_1.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[4])))
    pytorch_mix.branch7x7dbl_2.conv = convert_conv(pytorch_mix.branch7x7dbl_2.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[5])))
    pytorch_mix.branch7x7dbl_2.bn = convert_bn(pytorch_mix.branch7x7dbl_2.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[5])))
    pytorch_mix.branch7x7dbl_3.conv = convert_conv(pytorch_mix.branch7x7dbl_3.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[6])))
    pytorch_mix.branch7x7dbl_3.bn = convert_bn(pytorch_mix.branch7x7dbl_3.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[6])))
    pytorch_mix.branch7x7dbl_4.conv = convert_conv(pytorch_mix.branch7x7dbl_4.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[7])))
    pytorch_mix.branch7x7dbl_4.bn = convert_bn(pytorch_mix.branch7x7dbl_4.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[7])))
    pytorch_mix.branch7x7dbl_5.conv = convert_conv(pytorch_mix.branch7x7dbl_5.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[8])))
    pytorch_mix.branch7x7dbl_5.bn = convert_bn(pytorch_mix.branch7x7dbl_5.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[8])))
    pytorch_mix.branch_pool.conv = convert_conv(pytorch_mix.branch_pool.conv, keras_model.get_layer("conv2d_"+str(keras_layer_list[9])))
    pytorch_mix.branch_pool.bn = convert_bn(pytorch_mix.branch_pool.bn, keras_model.get_layer("batch_normalization_"+str(keras_layer_list[9])))
    return pytorch_mix


def main(args):
    ptmodel = Inception3()
    keras_model = InceptionV3(weights=args.input_path, input_shape=(299, 299, 3), include_top=False)
    #keras_model = tf.keras.models.load_model(args.input_path)

    # Convert conv/bn weights
    ptmodel.Conv2d_1a_3x3.conv = convert_conv(ptmodel.Conv2d_1a_3x3.conv, keras_model.get_layer("conv2d"))
    ptmodel.Conv2d_1a_3x3.bn = convert_bn(ptmodel.Conv2d_1a_3x3.bn, keras_model.get_layer("batch_normalization"))
    ptmodel.Conv2d_2a_3x3.conv = convert_conv(ptmodel.Conv2d_2a_3x3.conv, keras_model.get_layer("conv2d_1"))
    ptmodel.Conv2d_2a_3x3.bn = convert_bn(ptmodel.Conv2d_2a_3x3.bn, keras_model.get_layer("batch_normalization_1"))
    ptmodel.Conv2d_2b_3x3.conv = convert_conv(ptmodel.Conv2d_2b_3x3.conv, keras_model.get_layer("conv2d_2"))
    ptmodel.Conv2d_2b_3x3.bn = convert_bn(ptmodel.Conv2d_2b_3x3.bn, keras_model.get_layer("batch_normalization_2"))
    ptmodel.Conv2d_3b_1x1.conv = convert_conv(ptmodel.Conv2d_3b_1x1.conv, keras_model.get_layer("conv2d_3"))
    ptmodel.Conv2d_3b_1x1.bn = convert_bn(ptmodel.Conv2d_3b_1x1.bn, keras_model.get_layer("batch_normalization_3"))
    ptmodel.Conv2d_4a_3x3.conv = convert_conv(ptmodel.Conv2d_4a_3x3.conv, keras_model.get_layer("conv2d_4"))
    ptmodel.Conv2d_4a_3x3.bn = convert_bn(ptmodel.Conv2d_4a_3x3.bn, keras_model.get_layer("batch_normalization_4"))

    #convert mix layers 5b/5c/5d
    ptmodel.Mixed_5b = convert_mix_5(ptmodel.Mixed_5b, keras_model,'5b')
    ptmodel.Mixed_5c = convert_mix_5(ptmodel.Mixed_5c, keras_model,'5c')
    ptmodel.Mixed_5d = convert_mix_5(ptmodel.Mixed_5d, keras_model,'5d')

    #convert mix layers 6a/6b/6c/6d/6e
    ptmodel.Mixed_6a = convert_mix_6(ptmodel.Mixed_6a, keras_model,'6a')
    ptmodel.Mixed_6b = convert_mix_6(ptmodel.Mixed_6b, keras_model,'6b')
    ptmodel.Mixed_6c = convert_mix_6(ptmodel.Mixed_6c, keras_model,'6c')
    ptmodel.Mixed_6d = convert_mix_6(ptmodel.Mixed_6d, keras_model,'6d')
    ptmodel.Mixed_6e = convert_mix_6(ptmodel.Mixed_6e, keras_model,'6e')

    #convert mix layers 7a/7b/7c
    ptmodel.Mixed_7a.branch3x3_1.conv = convert_conv(ptmodel.Mixed_7a.branch3x3_1.conv, keras_model.get_layer("conv2d_70"))
    ptmodel.Mixed_7a.branch3x3_1.bn = convert_bn(ptmodel.Mixed_7a.branch3x3_1.bn, keras_model.get_layer("batch_normalization_70"))
    ptmodel.Mixed_7a.branch3x3_2.conv = convert_conv(ptmodel.Mixed_7a.branch3x3_2.conv, keras_model.get_layer("conv2d_71"))
    ptmodel.Mixed_7a.branch3x3_2.bn = convert_bn(ptmodel.Mixed_7a.branch3x3_2.bn, keras_model.get_layer("batch_normalization_71"))
    ptmodel.Mixed_7a.branch7x7x3_1.conv = convert_conv(ptmodel.Mixed_7a.branch7x7x3_1.conv, keras_model.get_layer("conv2d_72"))
    ptmodel.Mixed_7a.branch7x7x3_1.bn = convert_bn(ptmodel.Mixed_7a.branch7x7x3_1.bn, keras_model.get_layer("batch_normalization_72"))
    ptmodel.Mixed_7a.branch7x7x3_2.conv = convert_conv(ptmodel.Mixed_7a.branch7x7x3_2.conv, keras_model.get_layer("conv2d_73"))
    ptmodel.Mixed_7a.branch7x7x3_2.bn = convert_bn(ptmodel.Mixed_7a.branch7x7x3_2.bn, keras_model.get_layer("batch_normalization_73"))
    ptmodel.Mixed_7a.branch7x7x3_3.conv = convert_conv(ptmodel.Mixed_7a.branch7x7x3_3.conv, keras_model.get_layer("conv2d_74"))
    ptmodel.Mixed_7a.branch7x7x3_3.bn = convert_bn(ptmodel.Mixed_7a.branch7x7x3_3.bn, keras_model.get_layer("batch_normalization_74"))
    ptmodel.Mixed_7a.branch7x7x3_4.conv = convert_conv(ptmodel.Mixed_7a.branch7x7x3_4.conv, keras_model.get_layer("conv2d_75"))
    ptmodel.Mixed_7a.branch7x7x3_4.bn = convert_bn(ptmodel.Mixed_7a.branch7x7x3_4.bn, keras_model.get_layer("batch_normalization_75"))

    #7b/7c
    ptmodel.Mixed_7b.branch1x1.conv = convert_conv(ptmodel.Mixed_7b.branch1x1.conv, keras_model.get_layer("conv2d_76"))
    ptmodel.Mixed_7b.branch1x1.bn = convert_bn(ptmodel.Mixed_7b.branch1x1.bn, keras_model.get_layer("batch_normalization_76"))
    ptmodel.Mixed_7b.branch3x3_1.conv = convert_conv(ptmodel.Mixed_7b.branch3x3_1.conv, keras_model.get_layer("conv2d_77"))
    ptmodel.Mixed_7b.branch3x3_1.bn = convert_bn(ptmodel.Mixed_7b.branch3x3_1.bn, keras_model.get_layer("batch_normalization_77"))
    ptmodel.Mixed_7b.branch3x3_2a.conv = convert_conv(ptmodel.Mixed_7b.branch3x3_2a.conv, keras_model.get_layer("conv2d_78"))
    ptmodel.Mixed_7b.branch3x3_2a.bn = convert_bn(ptmodel.Mixed_7b.branch3x3_2a.bn, keras_model.get_layer("batch_normalization_78"))
    ptmodel.Mixed_7b.branch3x3_2b.conv = convert_conv(ptmodel.Mixed_7b.branch3x3_2b.conv, keras_model.get_layer("conv2d_79"))
    ptmodel.Mixed_7b.branch3x3_2b.bn = convert_bn(ptmodel.Mixed_7b.branch3x3_2b.bn, keras_model.get_layer("batch_normalization_79"))
    ptmodel.Mixed_7b.branch3x3dbl_1.conv = convert_conv(ptmodel.Mixed_7b.branch3x3dbl_1.conv, keras_model.get_layer("conv2d_80"))
    ptmodel.Mixed_7b.branch3x3dbl_1.bn = convert_bn(ptmodel.Mixed_7b.branch3x3dbl_1.bn, keras_model.get_layer("batch_normalization_80"))
    ptmodel.Mixed_7b.branch3x3dbl_2.conv = convert_conv(ptmodel.Mixed_7b.branch3x3dbl_2.conv, keras_model.get_layer("conv2d_81"))
    ptmodel.Mixed_7b.branch3x3dbl_2.bn = convert_bn(ptmodel.Mixed_7b.branch3x3dbl_2.bn, keras_model.get_layer("batch_normalization_81"))
    ptmodel.Mixed_7b.branch3x3dbl_3a.conv = convert_conv(ptmodel.Mixed_7b.branch3x3dbl_3a.conv, keras_model.get_layer("conv2d_82"))
    ptmodel.Mixed_7b.branch3x3dbl_3a.bn = convert_bn(ptmodel.Mixed_7b.branch3x3dbl_3a.bn, keras_model.get_layer("batch_normalization_82"))
    ptmodel.Mixed_7b.branch3x3dbl_3b.conv = convert_conv(ptmodel.Mixed_7b.branch3x3dbl_3b.conv, keras_model.get_layer("conv2d_83"))
    ptmodel.Mixed_7b.branch3x3dbl_3b.bn = convert_bn(ptmodel.Mixed_7b.branch3x3dbl_3b.bn, keras_model.get_layer("batch_normalization_83"))
    ptmodel.Mixed_7b.branch_pool.conv = convert_conv(ptmodel.Mixed_7b.branch_pool.conv, keras_model.get_layer("conv2d_84"))
    ptmodel.Mixed_7b.branch_pool.bn = convert_bn(ptmodel.Mixed_7b.branch_pool.bn, keras_model.get_layer("batch_normalization_84"))

    ptmodel.Mixed_7c.branch1x1.conv = convert_conv(ptmodel.Mixed_7c.branch1x1.conv, keras_model.get_layer("conv2d_85"))
    ptmodel.Mixed_7c.branch1x1.bn = convert_bn(ptmodel.Mixed_7c.branch1x1.bn, keras_model.get_layer("batch_normalization_85"))
    ptmodel.Mixed_7c.branch3x3_1.conv = convert_conv(ptmodel.Mixed_7c.branch3x3_1.conv, keras_model.get_layer("conv2d_86"))
    ptmodel.Mixed_7c.branch3x3_1.bn = convert_bn(ptmodel.Mixed_7c.branch3x3_1.bn, keras_model.get_layer("batch_normalization_86"))
    ptmodel.Mixed_7c.branch3x3_2a.conv = convert_conv(ptmodel.Mixed_7c.branch3x3_2a.conv, keras_model.get_layer("conv2d_87"))
    ptmodel.Mixed_7c.branch3x3_2a.bn = convert_bn(ptmodel.Mixed_7c.branch3x3_2a.bn, keras_model.get_layer("batch_normalization_87"))
    ptmodel.Mixed_7c.branch3x3_2b.conv = convert_conv(ptmodel.Mixed_7c.branch3x3_2b.conv, keras_model.get_layer("conv2d_88"))
    ptmodel.Mixed_7c.branch3x3_2b.bn = convert_bn(ptmodel.Mixed_7c.branch3x3_2b.bn, keras_model.get_layer("batch_normalization_88"))
    ptmodel.Mixed_7c.branch3x3dbl_1.conv = convert_conv(ptmodel.Mixed_7c.branch3x3dbl_1.conv, keras_model.get_layer("conv2d_89"))
    ptmodel.Mixed_7c.branch3x3dbl_1.bn = convert_bn(ptmodel.Mixed_7c.branch3x3dbl_1.bn, keras_model.get_layer("batch_normalization_89"))
    ptmodel.Mixed_7c.branch3x3dbl_2.conv = convert_conv(ptmodel.Mixed_7c.branch3x3dbl_2.conv, keras_model.get_layer("conv2d_90"))
    ptmodel.Mixed_7c.branch3x3dbl_2.bn = convert_bn(ptmodel.Mixed_7c.branch3x3dbl_2.bn, keras_model.get_layer("batch_normalization_90"))
    ptmodel.Mixed_7c.branch3x3dbl_3a.conv = convert_conv(ptmodel.Mixed_7c.branch3x3dbl_3a.conv, keras_model.get_layer("conv2d_91"))
    ptmodel.Mixed_7c.branch3x3dbl_3a.bn = convert_bn(ptmodel.Mixed_7c.branch3x3dbl_3a.bn, keras_model.get_layer("batch_normalization_91"))
    ptmodel.Mixed_7c.branch3x3dbl_3b.conv = convert_conv(ptmodel.Mixed_7c.branch3x3dbl_3b.conv, keras_model.get_layer("conv2d_92"))
    ptmodel.Mixed_7c.branch3x3dbl_3b.bn = convert_bn(ptmodel.Mixed_7c.branch3x3dbl_3b.bn, keras_model.get_layer("batch_normalization_92"))
    ptmodel.Mixed_7c.branch_pool.conv = convert_conv(ptmodel.Mixed_7c.branch_pool.conv, keras_model.get_layer("conv2d_93"))
    ptmodel.Mixed_7c.branch_pool.bn = convert_bn(ptmodel.Mixed_7c.branch_pool.bn, keras_model.get_layer("batch_normalization_93"))

    #ptmodel = torch.nn.Sequential(*(list(ptmodel.children())[:-3]))
    # Test converted model
    x = np.random.rand(100, 299, 299, 3)
    x_pt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).float()
    print(x_pt.shape)
    

    ptmodel.eval()
    with torch.no_grad():
        outputs_pt = ptmodel(x_pt)
        print(outputs_pt.numpy().shape)
        outputs_pt = np.transpose(outputs_pt.numpy(), (0, 2, 3, 1))

    x_tf = tf.convert_to_tensor(x)
    outputs_tf = keras_model.predict(x_tf)
    #outputs_tf = outputs_tf.numpy()
    print(outputs_tf.shape)
    print(f"Are the outputs all close (absolute tolerance = 1e-02)? {np.allclose(outputs_tf, outputs_pt, atol=1e-02)}")
    print("Pytorch output")
    print(outputs_pt[1, :30,:,:])
    print("Tensoflow Keras output")
    print(outputs_tf[1, :30,:,:])

    # Saving model
    torch.save(ptmodel.module.state_dict(), args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the original RadImageNet-ResNet50_notop.h5 file.")
    parser.add_argument("--output_path", help="Path to save the converted .pth file.")
    args = parser.parse_args()

    main(args)