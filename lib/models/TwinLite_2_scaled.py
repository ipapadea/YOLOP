import torch
import torch.nn as nn

import torch.nn.functional as f
from torch.nn import Module, Conv2d, Parameter, Softmax
from efficientFCN.encoding.models.efficientFCN import MultiHGDecoderTwinLiteNet2, MultiHGDModuleTwinLiteNetv2, \
    MHGDTwinLiteNet2Scaled, MultiHGDModuleTwinLiteNetv2Scaled, MultiHGDecoderTwinLiteNet2HalfCodewords


class PAM_Module_output(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, out_dim):
        super(PAM_Module_output, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        self.out_conv = Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        out = self.out_conv(out)  # Change the channel dimension
        return out


class CAM_Module_output(Module):
    """ Channel attention module"""

    def __init__(self, in_dim, out_dim):
        super(CAM_Module_output, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.out_conv = Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        out = self.out_conv(out)  # Change the channel dimension
        return out


class Efficient_PAM_Module(Module):
    """ Position attention module"""

    def __init__(self, in_dim, out_dim):
        super(Efficient_PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        self.out_conv = Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)  # .permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        proj_key = f.softmax(proj_key, dim=2)
        proj_query = f.softmax(proj_query, dim=1)
        context = proj_key @ proj_value.transpose(1, 2)
        out = (context.transpose(1, 2) @ proj_query).reshape(m_batchsize, C, height, width)
        out = self.gamma * out + x
        out = self.out_conv(out)  # Change the channel dimension
        return out


class Efficient_CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim, out_dim):
        super(Efficient_CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.out_conv = Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_value = x.view(m_batchsize, C, -1)
        proj_key = f.softmax(proj_key, dim=2)
        proj_query = f.softmax(proj_query, dim=1)
        context = torch.bmm(proj_key, proj_value)  # [1, 240] , [1, 1024]
        out = torch.bmm(context, proj_query)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        out = self.out_conv(out)  # Change the channel dimension

        return out


class UPx2_scaled(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, scale=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        self.deconv = nn.ConvTranspose2d(int(nIn * scale), nOut, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

    def fuseforward(self, input):
        output = self.deconv(input)
        output = self.act(output)
        return output


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        # self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        # self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

    def fuseforward(self, input):
        output = self.conv(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        # print(nIn, nOut, (kSize, kSize))
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.nOut = nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        # print("bf bn :",input.size(),self.nOut)
        output = self.bn(input)
        # print("after bn :",output.size())
        output = self.act(output)
        # print("after act :",output.size())
        return output


class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = max(int(nOut / 5), 1)
        n1 = max(nOut - 4 * n, 1)
        # print(nIn,n,n1,"--")
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        # print("nOut bf :",nOut)
        self.bn = BR(nOut)
        # print("nOut at :",self.bn.size())
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        # print(d1.size(),add1.size(),add2.size(),add3.size(),add4.size())

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # print("combine :",combine.size())
        # if residual version
        if self.add:
            # print("add :",combine.size())
            combine = input + combine
        # print(combine.size(),"-----------------")
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet2_Encoder_scaled(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''

    def __init__(self, p=5, q=3, scale=1.0):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(3, int(16 * scale), 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = CBR(int(16 * scale) + 3, int(19 * scale), 3)
        # self.b1 = CBR(int((16+3) * scale), int(19 * scale), 3)

        # self.level2_0 = DownSamplerB(int(16 * scale) + 3, int(64 * scale))
        self.level2_0 = DownSamplerB(int((16 + 3) * scale), int(64 * scale))

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(int(64 * scale), int(64 * scale)))
        self.b2 = CBR(int(128 * scale) + 3, int(131 * scale), 3)
        # self.b2 = CBR(int((128+3) * scale) , int(131 * scale), 3)
        self.level3_0 = DownSamplerB(int((128 + 3) * scale), int(128 * scale))
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(int(128 * scale), int(128 * scale)))
        # self.b3 = CBR(256,32,3) # to evgala gia na mpei o decoder
        self.decoder = MHGDTwinLiteNet2Scaled(scale=scale, in_channels=int(64 * scale), out_channels=int(32 * scale),
                                              num_center=int(64 * scale), norm_layer=nn.BatchNorm2d)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        # print("input shape: ", input.shape)
        output0 = self.level1(input)
        # print("output0 shape: ", output0.shape)
        inp1 = self.sample1(input)
        # print("inp1 shape: ", inp1.shape)
        inp2 = self.sample2(input)
        # print("inp2 shape: ", inp2.shape)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        # print("output0_cat shape: ", output0_cat.shape)

        output1_0 = self.level2_0(output0_cat)  # down-sampled
        # print("output1_0 shape: ", output1_0.shape)

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        cat_ = torch.cat([output2_0, output2], 1)
        # print("output of cat_ shape: ", cat_.shape)

        # output2_cat = self.b3(cat_)
        # print("input of multihgd shape: ", cat_.shape)

        decoder_output = self.decoder((output0_cat, output1_cat, cat_))  # eksodos tou multiHGD

        return decoder_output  # cat_ # classifier


class TwinLiteNet_2_scaled(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, p=2, q=3, scale=1.0):
        super().__init__()
        self.encoder = ESPNet2_Encoder_scaled(p, q, scale)

        # self.up_1_1 = UPx2(32,16)
        # self.up_2_1 = UPx2(16,8)
        #
        # self.up_1_2 = UPx2(32,16)
        # self.up_2_2 = UPx2(16,8)
        #
        # self.classifier_1 = UPx2(8,2)
        # self.classifier_2 = UPx2(8,2)

        self.classifier_1 = UPx2_scaled(int(32 * scale), 2)
        self.classifier_2 = UPx2_scaled(int(32 * scale), 2)

    def forward(self, input):
        x = self.encoder(input)
        # x1=self.up_1_1(x[0])
        # x1=self.up_2_1(x1)
        # classifier1=self.classifier_1(x1)
        #
        #
        #
        # x2=self.up_1_2(x[1])
        # x2=self.up_2_2(x2)
        # classifier2=self.classifier_2(x2)

        classifier1 = self.classifier_1(x[0])
        classifier2 = self.classifier_2(x[1])

        return (classifier1, classifier2)

# Instantiate different versions of the model
# from utils import netParams
# small_model = TwinLiteNet_2_scaled(scale=0.5)
# total_paramters = netParams(small_model)
# print('small_model network parameters: ' + str(total_paramters))
#
# nano_model = TwinLiteNet_2_scaled(scale=0.25)
# total_paramters = netParams(nano_model)
# print('nano_model network parameters: ' + str(total_paramters))
#
# medium_model = TwinLiteNet_2_scaled(scale=1.0)
# total_paramters = netParams(medium_model)
# print('medium_model network parameters: ' + str(total_paramters))
#
# large_model = TwinLiteNet_2_scaled(scale=2.0)
# total_paramters = netParams(large_model)
# print('large_model network parameters: ' + str(total_paramters))
from torchsummary import summary
import torch

# Instantiate your model
model = TwinLiteNet_2_scaled(scale=1.0)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Run a forward pass to inspect the outputs
def hook_fn(module, input, output):
    module_name = module.__class__.__name__
    if isinstance(output, tuple):  # If the output is a tuple (multi-output layer)
        for i, out in enumerate(output):
            print(f"Layer: {module_name}, Output {i}: Shape {out.shape}")
    else:
        print(f"Layer: {module_name}, Output Shape: {output.shape}")

# Function to register hooks
def register_hooks(model):
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)):
            layer.register_forward_hook(hook_fn)

register_hooks(model)

# Create a dummy input
x = torch.randn(1, 3, 360, 640).to(device)

# Forward pass
with torch.no_grad():
    model(x)  # This will trigger