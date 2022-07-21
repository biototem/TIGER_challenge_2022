'''
这个是根据我复现的sfcn-opi论文，修改后用于tiger_game
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rev_blocks import RevGroupBlock

import albumentations as albu

def ConvBnAct(in_channel, out_channel, kernel_size, stride, pad, act=nn.Identity()): # 128, 2, 1, 1,0,act
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad),
        nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.1),
        act
    )


def ConvBn(in_channel, out_channel, kernel_size, stride, pad, act=nn.Identity()):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad),
        nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.1),
    )

class RevBlockC(nn.Module):  # Module 1 里面的Residual Block    RevBlockC不进行下采样
    def __init__(self, in_channel, out_channel, stride, act, **kwargs):
        super().__init__()
        assert in_channel == out_channel  # 输入通道应该等于输出通道
        assert stride == 1
        temp_channel = in_channel // 1  # 一个中间变量
        self.conv1 = ConvBnAct(in_channel, temp_channel, kernel_size=3, stride=1, pad=1, act=act)
        self.conv2 = ConvBn(temp_channel, out_channel, kernel_size=3, stride=1, pad=1)
        # 上面经过两个卷积层，图片的通道数和大小都不会变化。

        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x_shortcut = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = x_shortcut + y  # 残差链接

        # y = self.relu(y)

        y_final = F.relu(y,inplace=True)

        return y_final

class ResBlockA(nn.Module):  # 在 Module 2中第一个Residual Block需要进行下采样一倍(降低分辨率)
    def __init__(self, in_channel, out_channel, stride, act):
        super().__init__()

        temp_ch = out_channel // 1  # 这一步的目的就是为了看着好看，输出通道的位置
        self.conv1 = ConvBnAct(in_channel, temp_ch, kernel_size=3, stride=stride, pad=1, act=act)  # 这里使用卷积进行下采样
        self.conv2 = ConvBn(temp_ch, out_channel, kernel_size=3, stride=1, pad=1)

        self.conv3 = ConvBn(in_channel, out_channel, kernel_size=3, stride=stride, pad=1)     #  对x也进行卷积，为了下采样

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        x_shortcut = self.conv3(x)
        y_final = y + x_shortcut  # 这里的y就是论文中的Fi，x_shortcut就是Ws
        y_final = F.relu(y_final,inplace=True)
        return y_final

class MainNet(nn.Module):
    model_id = 1
    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU(0.02, inplace=True)
        # act = nn.ReLU(inplace=True)

# ==============================================   第一阶段 共享层和粗检测==================================================
        self._bm1 = nn.ModuleList()  # 第一阶段，共享底层
        self.conv1 = ConvBnAct(3, 32, 3, 1, 1, act)  # 首先经过一个卷积层
        self.rvb1 = RevGroupBlock(32, 32, 1, act, RevBlockC, 9)  # 这里是Module1  RevBlockC不进行下采样

        self.rb2 = ResBlockA(32, 64, 2, act)  # Module2 的开始部分，进行分辨率的降低，RevBlockA 进行下采样
        self.rvb2 = RevGroupBlock(64, 64, 1, act, RevBlockC, 8)  # Module2 的剩下8个块  RevBlockC不进行下采样

        self.rb3 = ResBlockA(64, 128, 2, act)  # Module 3 的开始部分，进行分辨率的降低，RevBlockA 进行下采样
        self.rvb3 = RevGroupBlock(128, 128, 1, act, RevBlockC, 8)  # Module3 的剩下8个块  RevBlockC不进行下采样

        self.conv2 = nn.Conv2d(128, 2, 1, 1, 0)
        self.deconv1 = nn.ConvTranspose2d(2, 2, 3, 2, 1, 1)
        self.conv3 = nn.Conv2d(64, 2, 1, 1, 0)
        self.deconv2 = nn.ConvTranspose2d(2, 2, 3, 2, 1, 1)

        self._bm1.extend([self.conv1, self.rvb1, self.rb2, self.rvb2, self.rb3, self.rvb3,self.conv2, self.deconv1, self.conv3, self.deconv2])

# ==============================================   第二阶段 细检测==================================================

        self._bm2 = nn.ModuleList()

        self.rvb4 = RevGroupBlock(128, 128, 1, act, RevBlockC, 9)
        self.conv4 = nn.Conv2d(128, 2, 1, 1, 0)
        self.deconv4 = nn.ConvTranspose2d(2, 2, 3, 2, 1, 1)
        self.deconv5 = nn.ConvTranspose2d(2, 2, 3, 2, 1, 1)

        self._bm2.extend([self.rvb4,self.conv4,self.deconv4,self.deconv5])

 # ================================================= 训练策略===========================================

        self.enabled_b2_branch = False  # 开启假阳性抑制分支
        self.is_freeze_seg1 = True      # 是否冻结共享层和粗检测层
        self.is_freeze_seg2 = True      # 是否冻结假阳性抑制层

    def set_freeze_seg1(self, b):
        assert isinstance(b, bool)
        self.is_freeze_seg1 = b
        self._bm1.train(not b)              # 如果b = True，(共享层)就不进行训练
        for p in self._bm1.parameters():    # 如果b = True，(共享层)不进行参数更新
            p.requires_grad = not b

    def set_freeze_seg2(self, b):
        assert isinstance(b, bool)
        self.is_freeze_seg2 = b
        self._bm2.train(not b)             # 如果b = True，(检测分支)就不进行训练
        for p in self._bm2.parameters():   # 如果b = True，(检测分支)不进行参数更新
            p.requires_grad = not b


    def seg1_state_dict(self):
        return dict(self._bm1.state_dict())                 # 返回(共享层)的训练参数

    def load_seg1_state_dict(self, *args, **kwargs):
        self._bm1.load_state_dict(*args, **kwargs)          # 加载(共享层)的训练参数

    def seg2_state_dict(self):
        return dict(self._bm2.state_dict())                 # 返回(检测分支)的训练参数

    def load_seg2_state_dict(self, *args, **kwargs):
        self._bm2.load_state_dict(*args, **kwargs)          # 加载(检测分支)的训练参数


    def forward(self, x):

# ========================= 第一阶段，共享层和粗检测层=======================

        y = self.conv1(x)
        y = self.rvb1(y)
        y = self.rb2(y)
        y = self.rvb2(y)                     # modul2结束
        y_2 = y                              # y_2用于检测分支

        y = self.rb3(y)
        y = self.rvb3(y)                     # module3结束
        y_3 = y                             # y_3用于分类分支

# ========================= 检测分支 ============================
        y_det, y_cla = None,None

        y_m2 = self.conv3(y_2)
        y = self.conv2(y)
        y = self.deconv1(y)
        y_det = self.deconv2(y + y_m2)

# ========================= 假阳性抑制分支 =============================
        if self.enabled_b2_branch:              # 开启核分类分支
            y1 = self.rvb4(y_3)
            y2 = self.conv4(y1)
            y3 = self.deconv4(y2)
            y_cla = self.deconv5(y3)

        return y_det, y_cla    # y_det 2个通道，bg和细胞核， 这里的y_cla是进一步假阳性抑制 所以也是2个通道、


if __name__ == '__main__':
    a = torch.zeros(8, 3, 64,64).cuda(0)
    # a = torch.zeros(2, 512, 512, 3).cuda(0).permute(0, 3, 1, 2)
    BATCH_SIZE = 3
    net = MainNet().cuda(0)
    net.enabled_b2_branch = True
    transform = albu.Compose([
        albu.Resize(64, 64)  # 进行图片的变形
    ])  # 定义数据增强方式
    # train_dataset = DatasetReader(r'F:\xie_deep_learning\New_data\\','train',use_blance=True,augment=True,transforms = transform)
    # train_loader = DataLoader(dataset=train_dataset,batch_size=3,shuffle=True,num_workers=0)
    # image, mask_det,mask_cls = next(iter(train_loader))
    # image = image.cuda(0)
    out_det,out_cla = net(a)
    print(out_cla.shape)
    print(out_det.shape)
