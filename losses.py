import torch
from torch import fft as fft
import torchvision
from globals import TRAINING_CROP_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(StyleLoss, self).__init__()
        blocks = []
        VGG19 = torchvision.models.vgg19(pretrained=True).to(device)
        blocks.append(VGG19.features[:4].eval())
        blocks.append(VGG19.features[4:9].eval())
        blocks.append(VGG19.features[9:18].eval())
        blocks.append(VGG19.features[18:27].eval())
        blocks.append(VGG19.features[18:27].eval())
        # blocks.append(torchvision.models.vgg19(pretrained=True).features[27:36].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).eval().features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]




class FFTFeatureLoss(torch.nn.Module):
    def __init__(self):
        super(FFTFeatureLoss, self).__init__()
        self.vgg = VGG19().to(device)
        self.l1_loss = torch.nn.L1Loss()
        self.weights = [1.0,1.0/4,1.0/8,1.0/16,1.0/32]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
    def forward(self,pred,gt):
        pred = (pred-self.mean) / self.std
        gt = (gt-self.mean) / self.std
        pred_feat = self.vgg(pred)
        gt_feat = self.vgg(gt)
        pred_fft_feat = [fft.rfftn(feat,dim=(2,3),norm='ortho') for feat in pred_feat]
        gt_fft_feat = [fft.rfftn(feat,dim=(2,3),norm='ortho') for feat in gt_feat]
        loss = 0        
        for i in range(len(pred_fft_feat)):
            feat_loss = self.l1_loss(pred_fft_feat[i],gt_fft_feat[i])
            loss+=self.weights[i]*feat_loss
        return loss

            
