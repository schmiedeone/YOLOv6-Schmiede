import torch
from torch import nn
from yolov6.layers.common import RepBlock, RepVGGBlock, BottleRep, BepC3, SimConv, Transpose, BiFusion

# _QUANT=False
class RepPANNeck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[5],
            block=block
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[6],
            block=block
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=block
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=block
        )

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[5],
            out_channels=channels_list[5],
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[6],
            out_channels=channels_list[6]
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6],
            out_channels=channels_list[7],
            kernel_size=3,
            stride=2
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8],
            out_channels=channels_list[9],
            kernel_size=3,
            stride=2
        )

    def upsample_enable_quant(self, num_bits, calib_method):
        print("Insert fakequant after upsample")
        # Insert fakequant after upsample op to build TensorRT engine
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.tensor_quant import QuantDescriptor
        conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)
        self.upsample_feat0_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
        self.upsample_feat1_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
        # global _QUANT
        self._QUANT = True

    def forward(self, input):

        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        if hasattr(self, '_QUANT') and self._QUANT is True:
            upsample_feat0 = self.upsample_feat0_quant(upsample_feat0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        if hasattr(self, '_QUANT') and self._QUANT is True:
            upsample_feat1 = self.upsample_feat1_quant(upsample_feat1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class RepBiFPANNeck(nn.Module):
    """RepBiFPANNeck Module
    """
    # [64, 128, 256, 512, 1024] 
    # [256, 128, 128, 256, 256, 512]
    
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None  

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[5]], # 512, 256
            out_channels=channels_list[5], # 256
        )
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            block=block
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[5], channels_list[6]], # 256, 128
            out_channels=channels_list[6], # 128
        )
      
        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            block=block
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            block=block
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            block=block
        )
        

    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class RepPANNeck6(nn.Module):
    """RepPANNeck+P6 Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """
    # [64, 128, 256, 512, 768, 1024]   
    # [512, 256, 128, 256, 512, 1024]
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = RepBlock(
            in_channels=channels_list[4] + channels_list[6], # 768 + 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            block=block
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7] # 256
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[7], # 512 + 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            block=block
        )

        self.reduce_layer2 = SimConv(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.upsample2 = Transpose(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8] # 128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[8], # 256 + 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            block=block
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            block=block
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = RepBlock(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            block=block
        )

        self.downsample0 = SimConv(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = RepBlock(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            block=block
        )
        

    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        upsample_feat2 = self.upsample2(fpn_out2)
        f_concat_layer2 = torch.cat([upsample_feat2, x3], 1)
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class RepBiFPANNeck6(nn.Module):
    """RepBiFPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]   
    # [512, 256, 128, 256, 512, 1024]

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[4], channels_list[6]], # 768, 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = RepBlock(
            in_channels=channels_list[6], # 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            block=block
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[3], channels_list[7]], # 512, 256
            out_channels=channels_list[7], # 256
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            block=block
        )

        self.reduce_layer2 = SimConv(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[2], channels_list[8]], # 256, 128
            out_channels=channels_list[8], # 128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            block=block
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            block=block
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = RepBlock(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            block=block
        )

        self.downsample0 = SimConv(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = RepBlock(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            block=block
        )
        

    def forward(self, input):

        (x4, x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        f_concat_layer2 = self.Bifusion2([fpn_out2, x3, x4])
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepPANNeck(nn.Module):
    """
    CSPRepPANNeck module.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.Rep_p4 = BepC3(
            in_channels=channels_list[3] + channels_list[5], # 512 + 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        self.Rep_p3 = BepC3(
            in_channels=channels_list[2] + channels_list[6], # 256 + 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.Rep_n3 = BepC3(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.Rep_n4 = BepC3(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6] # 128
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

    def forward(self, input):

        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepBiFPANNeck(nn.Module): 
    """
    CSPRepBiFPANNeck module. 
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        
        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4], # 1024 
            out_channels=channels_list[5], # 256 
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[5]], # 512, 256
            out_channels=channels_list[5], # 256
        )

        self.Rep_p4 = BepC3(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128 
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[5], channels_list[6]], # 256, 128
            out_channels=channels_list[6], # 128
        )

        self.Rep_p3 = BepC3(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128 
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6], # 128 
            out_channels=channels_list[7], # 128 
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = BepC3(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128 
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8], # 256 
            out_channels=channels_list[9], # 256 
            kernel_size=3,
            stride=2
        )


        self.Rep_n4 = BepC3(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256 
            out_channels=channels_list[10], # 512 
            n=num_repeats[8],
            e=csp_e,
            block=block
        )
 
        
    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepPANNeck_P6(nn.Module):
    """CSPRepPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]   
    # [512, 256, 128, 256, 512, 1024]
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = BepC3(
            in_channels=channels_list[4] + channels_list[6], # 768 + 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7] # 256
        )

        self.Rep_p4 = BepC3(
            in_channels=channels_list[3] + channels_list[7], # 512 + 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.reduce_layer2 = SimConv(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.upsample2 = Transpose(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8] # 128
        )

        self.Rep_p3 = BepC3(
            in_channels=channels_list[2] + channels_list[8], # 256 + 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = BepC3(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            e=csp_e,
            block=block
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = BepC3(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            e=csp_e,
            block=block
        )

        self.downsample0 = SimConv(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = BepC3(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            e=csp_e,
            block=block
        )
        

    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        upsample_feat2 = self.upsample2(fpn_out2)
        f_concat_layer2 = torch.cat([upsample_feat2, x3], 1)
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepBiFPANNeck_P6(nn.Module):
    """CSPRepBiFPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]   
    # [512, 256, 128, 256, 512, 1024]
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[5], # 1024
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[4], channels_list[6]], # 768, 512
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = BepC3(
            in_channels=channels_list[6], # 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[3], channels_list[7]], # 512, 256
            out_channels=channels_list[7], # 256
        )

        self.Rep_p4 = BepC3(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.reduce_layer2 = SimConv(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[2], channels_list[8]], # 256, 128
            out_channels=channels_list[8], # 128
        )

        self.Rep_p3 = BepC3(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = BepC3(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            e=csp_e,
            block=block
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = BepC3(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            e=csp_e,
            block=block
        )

        self.downsample0 = SimConv(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = BepC3(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            e=csp_e,
            block=block
        )
        

    def forward(self, input):

        (x4, x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        f_concat_layer2 = self.Bifusion2([fpn_out2, x3, x4])
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class RepBiFPANNeck_P2(nn.Module):
    """RepBiFPANNeck extended one level down, so P2/4 gets its own detection head.

    Outputs four levels at strides [4, 8, 16, 32] instead of three at [8, 16, 32].
    The top-down path gains a P3 -> P2 stage and the bottom-up path a P2 -> P3 stage;
    everything above P3 is unchanged from RepBiFPANNeck. Set fuse_P2=True on the
    backbone so x3 (the stride-4 feature) is available.

    Plain concat rather than BiFusion at the P2 level: BiFusion needs a
    higher-resolution feature to downsample into the fusion, and there is nothing
    below P2 in the backbone.

    channels_list (neck part, after the 5 backbone entries):
      [5]  P5 lateral / top-down P4 out       [10] bottom-up P3 out  -> detect
      [6]  P4 lateral / top-down P3 out       [11] P3 -> P4 downsample
      [7]  P3 lateral / P2 upsample           [12] bottom-up P4 out  -> detect
      [8]  top-down P2 out       -> detect    [13] P4 -> P5 downsample
      [9]  P2 -> P3 downsample                [14] bottom-up P5 out  -> detect

    num_repeats (neck part): [5] Rep_p4, [6] Rep_p3, [7] Rep_p2, [8] Rep_n2,
    [9] Rep_n3, [10] Rep_n4.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        # --- top-down: P5 -> P4 (unchanged) ---
        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1
        )
        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[5]],
            out_channels=channels_list[5],
        )
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[5],
            block=block
        )

        # --- top-down: P4 -> P3 (unchanged) ---
        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        )
        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[5], channels_list[6]],
            out_channels=channels_list[6],
        )
        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[6],
            block=block
        )

        # --- top-down: P3 -> P2 (new) ---
        # The lateral stays thin so the stride-4 transposed conv is cheap; the fusion
        # widens back out so the P2 head has capacity.
        self.reduce_layer2 = SimConv(
            in_channels=channels_list[6],
            out_channels=channels_list[7],
            kernel_size=1,
            stride=1
        )
        self.upsample2 = Transpose(
            in_channels=channels_list[7],
            out_channels=channels_list[7],
        )
        self.Rep_p2 = RepBlock(
            in_channels=channels_list[7] + channels_list[1],
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=block
        )

        # --- bottom-up: P2 -> P3 (new) ---
        self.downsample3 = SimConv(
            in_channels=channels_list[8],
            out_channels=channels_list[9],
            kernel_size=3,
            stride=2
        )
        self.Rep_n2 = RepBlock(
            in_channels=channels_list[9] + channels_list[7],
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=block
        )

        # --- bottom-up: P3 -> P4 ---
        self.downsample2 = SimConv(
            in_channels=channels_list[10],
            out_channels=channels_list[11],
            kernel_size=3,
            stride=2
        )
        self.Rep_n3 = RepBlock(
            in_channels=channels_list[11] + channels_list[6],
            out_channels=channels_list[12],
            n=num_repeats[9],
            block=block
        )

        # --- bottom-up: P4 -> P5 ---
        self.downsample1 = SimConv(
            in_channels=channels_list[12],
            out_channels=channels_list[13],
            kernel_size=3,
            stride=2
        )
        self.Rep_n4 = RepBlock(
            in_channels=channels_list[13] + channels_list[5],
            out_channels=channels_list[14],
            n=num_repeats[10],
            block=block
        )

    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        f_out1 = self.Rep_p3(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        up_feat2 = self.upsample2(fpn_out2)
        f_concat_layer2 = torch.cat([up_feat2, x3], 1)
        pan_out3 = self.Rep_p2(f_concat_layer2)          # P2/4

        down_feat2 = self.downsample3(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n2(p_concat_layer2)          # P3/8

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)          # P4/16

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer0)          # P5/32

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs
