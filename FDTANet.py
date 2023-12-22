#This is an incomplete version, we will make all the code publicly available later


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arch_utils import LayerNorm2d
import torchvision.models as models


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class FourierUnit(nn.Module):
    def __init__(self, in_channels, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels * 4, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.norm = LayerNorm2d(in_channels * 4)
        self.sg = SimpleGate()
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.sg(self.norm(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output
    
class SpectralTransform(nn.Module):

    def __init__(self, in_channels, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=1, groups=1, bias=False),
            LayerNorm2d(in_channels*2),
            SimpleGate()
        )
        self.fu = FourierUnit(in_channels, **fu_kwargs)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output

class FFB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=1, bias=False),
            LayerNorm2d(in_channels),
            SimpleGate(),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1, groups=1, bias=False)
        )
        self.nonlo =  SpectralTransform(in_channels)
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, groups=1, bias=False)
    def forward(self, x):
        local = self.local(x)
        local = local + x
        nonlo = self.nonlo(x)
        out = torch.concatenate([local,nonlo], dim=1)
        out = self.conv(out)
        return out

    
class FFFN(nn.Module):
    def __init__(self, in_channels, FFN_Expand = 2, drop_out_rate=0., ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super().__init__()

        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
        ffn_channel = FFN_Expand * in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(in_channels)
        self.sg = SimpleGate()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.fft = nn.Parameter(torch.ones((in_channels, 1, 1)))
        self.conv3 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        

    def forward(self, inp):
        batch = inp.shape[0]
        x  = inp

        x = self.conv1(self.norm1(x))
        x = self.sg(x)
        x = self.conv2(x)

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = ffted * self.fft
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        
        output = self.sg(output)
        output = self.conv3(output) + x
        output = self.dropout1(output)

        return output * self.gamma + inp

# frequency domain-based gate
class FDGBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fffn = FFFN(c,FFN_Expand,drop_out_rate)
   
    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        out = self.fffn(y)

        return out


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


    
    
#frequency domain-based reweight 
class FDRBlock(nn.Module):
    def __init__(self, in_channels, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super().__init__()

        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
        self.sg = SimpleGate()
        self.fft = nn.Parameter(torch.ones((in_channels, 1, 1)))
        self.conv3 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        

    def forward(self, inp):
        batch = inp.shape[0]
        x  = inp

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = ffted * self.fft
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        output = self.sg(output)
        output = self.conv3(output)


        return output 


    
    
class ATBlock(nn.Module):
    def __init__(self, c,  drop_out_rate=0.):
        super().__init__()
        self.b1 = FDGBlock(c)
        self.b2 = FDGBlock(c)
        self.b3 = FDGBlock(c)

        self.fdr1 = FDRBlock(c)
        self.fdr2 = FDRBlock(c)
        self.fdr3 = FDRBlock(c)
    
    def forward(self, x):
        x1 = self.b1(x)
        x1_2 = self.fdr1(x1)


        x2 = self.b2(x+x1_2)
        x2_3 = self.fdr2(x2)
        
        x3 = self.b3(x+x1_2+x2_3)
        x3_ = self.fdr3(x3)


        return x1_2+x2_3+x3_+x
    

    


class FDTANet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.ffcb2 = FFB(width)
       

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[FDGBlock(chan) for _ in range(num)]
                    
                )
            )
          
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[FDGBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[ATBlock(chan) for _ in range(num)]
                )
            )
           

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
    


        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        

        x = self.ffcb2(x)
        x = self.ending(x)
        x = x + inp

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x




if __name__ == '__main__':
    

    x = torch.randn([2, 3, 256, 256])
    model = FDTANet()
    print("Total number of param  is ", sum(i.numel() for i in model.parameters()))
    t = model(x)
    print(t.shape)


    from thop import profile
    x3 =  torch.randn((1, 3, 256, 256))
    flops, params = profile(model, inputs=(x3, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    from ptflops import get_model_complexity_info
    FLOPS = 0
    inp_shape=(3,256,256)
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)
    #print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9



    print('mac', macs, params)

 
