import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
class SpatioTemporalConv(nn.Module):   
    '''时空卷积 
        filter:1*3*3
        filter:3*1*1
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        
        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels  \
        * out_channels)/(kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()   
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))  
        x = self.temporal_conv(x)                      
        return x
        
class Net(nn.Module):
    '''
    2+1D
    '''
    def __init__(self, frames=128):  
        super(Net, self).__init__()
        
        self.dense1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=(1,1,1), padding=[0,2,2]),                  
            nn.AvgPool3d(kernel_size=(1,2,2),stride=(1,2,2))
        )
            
        self.dense2 = nn.Sequential(
            SpatioTemporalConv(16,32,kernel_size=(3,3,3),stride=1,padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            SpatioTemporalConv(32,32,kernel_size=(3,3,3),stride=1,padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.AvgPool3d(kernel_size=(1,2,2),stride=(1,2,2))
        )
            
        self.dense3 =  nn.Sequential(
            SpatioTemporalConv(32,64,kernel_size=(3,3,3),stride=1,padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
 
            SpatioTemporalConv(64,64,kernel_size=(3,3,3),stride=1,padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.AvgPool3d(kernel_size=(1,2,2),stride=(1,2,2))
        )
            
        self.dense4 = nn.Sequential(
            SpatioTemporalConv(64,64,kernel_size=(3,3,3),stride=1,padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            SpatioTemporalConv(64,64,kernel_size=(3,3,3),stride=1,padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
            
        self.dense5 = nn.AdaptiveAvgPool3d((128,1,1))
            

        self.dense6 = nn.Conv3d(64,1,(1,1,1))
    
        
        self.correction = nn.Sequential(
            nn.Conv3d(32,1,kernel_size=(1,2,2),stride=(1,2,2)),
            nn.Conv3d(1,1,kernel_size=(1,2,2),stride=(1,2,2)),
            nn.Conv3d(1,1,kernel_size=(1,2,2),stride=(1,2,2)),
            nn.Conv3d(1,1,kernel_size=(1,2,2),stride=(1,2,2)),
        )
        
        ### 残差--->identity mapping恒等映射
        ### 使用Conv3D使tensor维度一致
        self.res_dense1 = nn.Conv3d(16,32,kernel_size=(1,2,2),stride=(1,2,2))
        self.res_dense2 = nn.Conv3d(32,64,kernel_size=(1,2,2),stride=(1,2,2))
        self.res_dense3 = nn.Conv3d(64,64,kernel_size=(1,1,1))
    def forward(self,x):
        x1 = self.dense1(x)                              #[,16,128,32,32]
        
        F_x = x1
        x2 = self.dense2(x1)                             #[,32,128,16,16]
        x2 = x2 + self.res_dense1(F_x)
        
        
        mid = self.correction(x2)                        #[,1,128,1,1]
        
        F_x = x2
        x3 = self.dense3(x2)                            #[,64,128,8,8]
        x3 = x3 + self.res_dense2(F_x)
        
        F_x = x3
        x4 = self.dense4(x3)                            #[,64,128,8,8]
        x4 = x4 + self.res_dense3(F_x)
        
        
        x5 = self.dense5(x4)                            #[,64,128,1,1]    
        x6 = self.dense6(x5)                            #[,1,128,1,1]

        rppg = x6.squeeze(1).squeeze(-1).squeeze(-1)    #[,128]
        mid = mid.squeeze(1).squeeze(-1).squeeze(-1)    #[,128]                                               
        return rppg,mid