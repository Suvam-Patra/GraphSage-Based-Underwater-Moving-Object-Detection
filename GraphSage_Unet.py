# -*- coding: utf-8 -*-
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torchvision.transforms as T
from torch.autograd import Variable
from torch.autograd import Function
import warnings
from PIL import Image
from random import randint
warnings.filterwarnings("ignore")
import random, os, cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchsummary import summary
from collections import OrderedDict
from torch import optim
from tqdm import tqdm
from sklearn import metrics
from sklearn.utils import shuffle
import torchvision
import torchvision.models as models
from torch_geometric.nn import SAGEConv
seed = 13
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# Change this to your project directory
base = '/content/UCN'

resnet = torchvision.models.resnet.resnet101(pretrained=True)


c = [randint(0, 1024) for i in range(0, 2047)]
m=[list(range(0,2047)),c]
m=torch.tensor(m,dtype=torch.int64)     #Random initialization of edges in graphs 
m=m.cuda(0)


class FileToTorchDataset(Dataset):
    def __init__(self, filePath,config):
        self.mu = config.mu
        self.std = config.std
        self.imagePaths1 = []
        self.imagePaths = []
        self.labels = []
        with open(filePath, "r") as f:
            for line in f:
                items = line.split(' ')
                self.imagePaths.append(base+'/CDnet_Underwater/original/'+items[0])
                #self.imagePaths.append(base+'/CDnet_Underwater/marine_snow/'+items[0])              #change the path to required dataset orignal or marine_snow
                self.imagePaths1.append(base+'/CDnet_Underwater/GT/'+items[1]) 
                self.labels.append(1)
        self.imagePaths,self.imagePaths1 ,self.labels = np.array(self.imagePaths),np.array(self.imagePaths1),np.array(self.labels)
        
    def __len__(self):
        return (len(self.labels))
    
    def __getitem__(self, i):
        img = cv2.imread(self.imagePaths[i], 1)
        img1 = cv2.imread(self.imagePaths1[i], 0)
        assert img is not None
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
        img1 = cv2.resize(img1, (224, 224), interpolation = cv2.INTER_CUBIC)
        img = img / 255.0
        img1 = np.expand_dims(img1, 0) / 255.0
        label = self.labels[i]
        img = torch.from_numpy(img).float()
        img1 = torch.from_numpy(img1).float()
        label = torch.from_numpy(np.asarray(label)).long()
        return img,img1 ,label
    
def createDataset(config):
    trainFile = base + '/label6.txt'  # File name of image details  ;  Use label6 & label 7 for original Images ::: label4 & label5 for enhanced (marine_snow
    testFile = base + '/label7.txt'
    BS = config.batchSize
    train = FileToTorchDataset(trainFile, config)
    test = FileToTorchDataset(testFile, config)  
    trainLoader = DataLoader(train, batch_size = BS, shuffle = True, num_workers=2)    
    testLoader  = DataLoader(test, batch_size = BS, shuffle = False, num_workers=2)
    return trainLoader, testLoader
    



class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = torchvision.models.resnet.resnet101(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.conv8=nn.Conv2d(32, 8, kernel_size=1)
        self.conv9=nn.Conv2d(8, 4, kernel_size=1)
        self.conv10=nn.Conv2d(4, 1, kernel_size=1)
        self.conv1 = SAGEConv(49,128)
        self.conv2 = SAGEConv(128,49)
        self.sig=nn.Sigmoid()
        self.up1=nn.Upsample(size=(9,9), mode='bilinear')
        self.up2=nn.Upsample(size=(14,14), mode='bilinear')
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, 32, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        op=torch.ones(len(x),2048,7,7).cuda(0)  
        #print(op.shape)
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x
         
        x = self.bridge(x)
        for q,p in enumerate(x):                        # Graph part, as gconv doesnt support batch processing, splitting batches individually and integrating them at last      
             p=torch.squeeze(p)
             p=p.reshape(2048,49)
             #p = self.conv1(p, m)
             #p = self.conv2(p, m)
             p=p.reshape(2048,7,7)
             p=torch.unsqueeze(p,0)
             op[q]=p
        x=op 
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        x1=self.conv8(x)
        x2=self.conv9(x1)
        x3=self.conv10(x2)
        x3=self.sig(x3)            #added sigmoid layer and some additional convolutional layers
        del pre_pools
        return x, x3

        

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config 
        self.model = UNetWithResnet50Encoder().cuda(config.gpu)
        for p in self.model.down_blocks.parameters():
            p.requires_grad = False                              # Encoder part is pre-trained and fully freezed
        if config.restore is not None:
            self.model.load_state_dict(torch.load(base + config.restore))
            print('Model loaded from '+ config.restore)
        if config.opt == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr = config.lr, weight_decay = config.l2)
        elif config.opt == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr = config.lr,momentum = 0.99, weight_decay = config.l2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 25, gamma = 0.1)
        self.loss=nn.BCELoss()
        self.trainLoader, self.testLoader = createDataset(config)
        self.trainHist = []
        self.testHist = []
        
    def train(self):
        if not os.path.exists(base + self.config.modelName):
            os.mkdir(base + self.config.modelName)
        for ep in tqdm(range(0, self.config.epochs)):
            runningLoss = 0
            acc=[]
            for i, data in enumerate(self.trainLoader):   
                if i % 100 == 0:
                    print('Batch Train {}'.format(i))
                imgs,gt = Variable(data[0].cuda(self.config.gpu)), Variable(data[1].cuda(self.config.gpu))
                imgs=imgs.permute(0,3,1,2)
                self.optimizer.zero_grad()                         
                avg_pool,y= self.model(imgs)
                loss=self.loss(y, gt)
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.item()
                correct_t = torch.sum(torch.round(y)==gt).item()
                total_t = gt.shape[0]*gt.shape[1]*gt.shape[2]*gt.shape[3]     
                acc_epc = 100*correct_t/total_t    
                acc.append(acc_epc)           
                del imgs, gt, loss, y
                torch.cuda.empty_cache()
            self.scheduler.step()  
            print("Loss-",runningLoss)    
            print("Epoch",ep,"Training Accuracy = ", sum(acc)/len(acc)) 
            if ep % 10== 0:
                torch.save(self.model.state_dict(), base + self.config.modelName +'/RES_' + str(ep) + '.pkl')
                print('Model saved to ' + self.config.modelName + '/RES' + str(ep) +'.pkl')
        torch.save(self.model.state_dict(), base + self.config.modelName +'/RES_18GCN.pkl')
        print('Model saved to ' + self.config.modelName + '/RES_w_50GCN.pkl')
        return np.array(self.trainHist), np.array(self.testHist)
    
    def test(self):
        runningLoss = 0
        acc=[]
        self.model.load_state_dict(torch.load("/content/UCN/Models/EN_fishy-2ON/RES_18GCN.pkl"))  # load the saved model here
        self.model.eval()
        rec=[]
        prec=[]
        for i,data in enumerate(self.testLoader):   
                if i % 5 == 0:
                  print('Batch Test {}'.format(i))
                imgs,gt = Variable(data[0].cuda(self.config.gpu)), Variable(data[1].cuda(self.config.gpu))
                imgs=imgs.permute(0,3,1,2)                       
                avg_pool,y= self.model(imgs)
                loss=self.loss(y, gt)
                #y=torch.round(y)
                #gt=torch.round(gt)
                #y=y*255
                #gt=gt*255
                runningLoss += loss.item()
                gt=torch.round(gt)
                y=torch.round(y)
                correct_t = torch.sum(y==gt).item()
                fp= ((y==1)&(gt==0))
                fn=((y==0)&(gt==1))
                tp=((y==0)&(gt==0))
                tn=((y==1)&(gt==1))
                pre=(correct_t ) / (correct_t + torch.sum(fp))
                r=100*((correct_t) / ( correct_t+ torch.sum(fn)))
                rec.append(r)
                prec.append(pre)
                total_t = gt.shape[0]*gt.shape[1]*gt.shape[2]*gt.shape[3]     
                acc_epc = 100*(correct_t/total_t)   
                acc.append(acc_epc)
                y=torch.squeeze(y)
                gt=torch.squeeze(gt)
                transform = T.ToPILImage()
                img = transform(torch.squeeze(y))
                cv2.imwrite(base +"/image_gen/"+str(i)+".jpg",np.array(img))
                img1=transform(torch.squeeze(gt))
                cv2.imwrite(base +"/image_gen/gt"+str(i)+".jpg",np.array(img1))
                '''
                y=torch.squeeze(y)
                h=y
                h.cuda(0)
                gt=torch.squeeze(gt)
                for ty in range(0,224):
                  for ui in range(0,224):
                    if ((torch.max(h)+torch.min(h))/2) <y[ty][ui]:
                      y[ty][ui]=1
                    elif ((torch.max(h)+torch.min(h))/2)>y[ty][ui]:
                      y[ty][ui]=0 
                    elif ((torch.max(h)+torch.min(h))/2)==y[ty][ui]:
                      y[ty][ui]= (torch.max(h)+torch.min(h))/2     
                transform = T.ToPILImage()
                img = transform(torch.squeeze(y))
                cv2.imwrite(base +"/image_gen/"+str(i)+".jpg",np.array(img))      
                img1=transform(torch.squeeze(gt))
                cv2.imwrite(base +"/image_gen/gt"+str(i)+".jpg",np.array(img1)) '''
                del imgs, gt, loss
                torch.cuda.empty_cache()
        print("Loss-",runningLoss)          
        print("Test Epoch -Accuracy = ", sum(acc)/len(acc))
        print("Test Epoch -Recall = ", sum(rec)/len(rec))
        print("Test Epoch -Precision = ", sum(prec)/len(prec))
        return(acc) 

class config():
    def __init__(self):
        self.gpu = 0
        self.seed = 13
        np.random.seed(self.seed)
        random.seed(self.seed) 
        torch.manual_seed(self.seed)
        self.mu = 0.449   
        self.std = 0.226  
        self.net = 'Global'
        self.modelName = "/Models/EN_fishy-2ON"   # directory where models need to be saved
        self.restore = None #"Models/fishy-2ON/RES_60.pkl"  # if want restore previous model and continue training
        self.batchSize = 1          #For testing and img genration change batch size to 'one'
        self.opt = 'SGD'  #SGD or Adam
        self.epochs = 100
        self.lr = 0.00085         #hyper-parameters
        self.l2 = 0.003
        self.bias = None 

conf = config()
wrap = Trainer(conf)
b1,a1 = wrap.train()     # for training
b1=wrap.test()           # for testing




