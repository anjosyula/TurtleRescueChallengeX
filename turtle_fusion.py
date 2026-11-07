"""RGB-depth extension for the pinned YOLO26n P2 architecture.

Training status belongs to each checkpoint, not this architecture definition.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseDown(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cin,3,2,1,groups=cin,bias=True),nn.SiLU(),
                         nn.Conv2d(cin,cout,1,bias=True),nn.SiLU())


class GeometryPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,8,3,2,1),nn.SiLU())
        self.blocks=nn.ModuleList([DepthwiseDown(8,16),DepthwiseDown(16,32),DepthwiseDown(32,64)])

    def forward(self,geometry):
        x=self.stem(geometry);features=[]
        for block in self.blocks:
            x=block(x);features.append(x)
        return features


class ConfidenceGatedFusion(nn.Module):
    def __init__(self,image_channels,depth_channels):
        super().__init__()
        self.projection=nn.Conv2d(depth_channels,image_channels,1)
        self.gate=nn.Sequential(nn.Conv2d(image_channels+depth_channels+1,16,1),nn.SiLU(),
                                nn.Conv2d(16,image_channels,1),nn.Sigmoid())
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)

    def forward(self,image,depth,quality):
        # Mean support, not max pooling: one point cannot make a cell fully valid.
        support=F.adaptive_avg_pool2d(quality,image.shape[-2:])
        gate=self.gate(torch.cat((image,depth,support),dim=1))
        return image+support*gate*self.projection(depth)


class RGBDepthP2(nn.Module):
    def __init__(self,detector):
        super().__init__()
        if detector.stride.tolist()!=[4.,8.,16.,32.]:raise ValueError('Expected P2/P3/P4/P5 detector')
        self.detector=detector
        # Infer the actual backbone widths; never assume stock class/head channel counts.
        channels={};handles=[]
        def capture(i):
            def hook(module,args,output):channels[i]=output.shape[1]
            return hook
        original_training=detector.training;detector.eval()
        for i in (2,4,6):handles.append(detector.model[i].register_forward_hook(capture(i)))
        try:
            with torch.no_grad():
                first=next(detector.parameters())
                detector(torch.zeros(1,3,128,128,device=first.device,dtype=first.dtype))
        finally:
            for handle in handles:handle.remove()
            detector.train(original_training)
        self.encoder=GeometryPyramid()
        self.fusions=nn.ModuleDict({str(i):ConfidenceGatedFusion(channels[i],d) for i,d in zip((2,4,6),(16,32,64))})

    def forward(self,image,geometry):
        if geometry.shape!=image.shape or geometry.shape[1]!=3:
            raise ValueError('Image and geometry must both be Bx3xHxW')
        finite_depth=torch.isfinite(geometry[:,0:1])
        geometry=torch.nan_to_num(geometry,nan=0.,posinf=0.,neginf=0.).clamp(0,1)
        validity=((geometry[:,1:2]>0.5)&finite_depth).to(geometry.dtype)
        geometry=torch.cat((geometry[:,0:1]*validity,validity,geometry[:,2:3]*validity),dim=1)
        quality=geometry[:,1:2]*geometry[:,2:3]
        features=self.encoder(geometry);saved=[];x=image
        for module in self.detector.model:
            if module.f!=-1:
                x=saved[module.f] if isinstance(module.f,int) else [x if j==-1 else saved[j] for j in module.f]
            x=module(x)
            if str(module.i) in self.fusions:
                index=(2,4,6).index(module.i)
                x=self.fusions[str(module.i)](x,features[index],quality)
            saved.append(x if module.i in self.detector.save else None)
        return x
