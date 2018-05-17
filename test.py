import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(generator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 62
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 62
            self.output_dim = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        return x

class WGAN_mnist(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.dk=args.dk
        self.c = 0.01                   # clipping value
        self.n_critic = 5               # the number of iterations of the critic per generator iteration
        if args.dk!=1:
            self.model_name="WGAN_DK_"+str(args.dk)
            print(self.model_name)

        # networks init
        self.D = discriminator(self.dataset)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.loss_fcn=nn.CrossEntropyLoss()
        if self.gpu_mode:
            self.D.cuda()


        # load dataset
        if self.dataset == 'mnist':
            self.data_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=True,
                                                         transform=transforms.Compose(
                                                             [transforms.ToTensor()])),
                                          batch_size=self.batch_size, shuffle=True)
            self.test_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=True,
                                                transform=transforms.Compose(
                                                    [transforms.ToTensor()])),
                                         batch_size=self.batch_size, shuffle=True)
        self.z_dim = 62

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        # self.train_hist['per_epoch_time'] = []
        # self.train_hist['total_time'] = []
        # self.train_hist['D_rate']=[]
        # self.train_hist['G_rate']=[]
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            # self.G.train()
            train_loss = 0
            train_acc = 0
            test_acc=0
            epoch_start_time = time.time()

            self.D.train()
            for iter, (x_, y_) in enumerate(self.data_loader):
                
                #这段代码表示怀疑
                # z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_,y_= Variable(x_.cuda()),Variable(y_.cuda())
                else:
                    x_,y= Variable(x_),Variable(y_)

                # update D network
                self.D_optimizer.zero_grad()

                out = self.D(x_)
                pre=torch.max(out,1)[1]     #预测的标记
                loss=self.loss_fcn(out,y_)  #计算的交叉熵损失
                loss_mean=loss.mean()
                loss_mean.backward()
                loss_mean.step()
                train_acc+=(pre==y_).sum()
            #迭代一轮完毕以后计算训练集的准确率
            acc=train_acc/len(self.data_loader)


            self.D.eval()
            for iter, (x_, y_) in enumerate(self.test_loader):
                if self.gpu_mode:
                    x_,y_= Variable(x_.cuda()),Variable(y_.cuda())
                else:
                    x_,y= Variable(x_),Variable(y_)   

                out = self.D(x_)
                pre=torch.max(out,1)[1]     #预测的标记                   
                test_acc+=(pre==y_).sum()  
            acc0=train_acc/len(self.data_loader)
            acc1=test_acc/len(self.test_loader)
            print(f"tran_acc is {acc0:.2f}, the test acc is {acc1:.2f}")
            self.train_hist['G_loss'].append(acc0)
            self.train_hist['G_rate'].append(acc1)

        self.save()

        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)
        #'将多个图片保存一张图片中'
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        '保存模型参数 并且保存train_hist的字典并且写入文件'
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        '载入模型的参数'
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))