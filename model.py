import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import h5py
import numpy as np

from networks.c3d import *
from networks.vlatt_hier_lstm import *

class Model(object):
    def __init__(self, config = None, train_set = None, test_set = None):
        # class that build, train and test model
        self.config = config
        self.train_set = train_set
        self.test_set = test_set

    def build(self):
        self.c3d = C3D()
        # freeze parameters in fc7 and fc8
        layer_idx = 0
        for child in self.c3d.children():
            if layer_idx >= 14: # layer fc7 and fc8
                for param in child.parameters():
                    param.requires_grad = False
            layer_idx += 1
        # load pre-trained c3d model
        self.c3d.load_state_dict(torch.load(str(self.config.c3d_model_dir) + '/c3d.pickle'))
        self.c3d.cuda()

        # variant lstm for spatial-temporal representation
        self.vlatt_hier_lstm = VLAttHierLstm(self.config.input_size, self.config.hidden_size,
                                             self.config.num_layers).cuda()

        # get scores
        self.fc1 = nn.Linear(self.config.hidden_size, 256).cuda()
        self.fc2 = nn.Linear(256, 1).cuda()

        self.model = nn.ModuleList([self.c3d, self.vlatt_hier_lstm, self.fc1, self.fc2])

        # minimal squared error loss
        self.mse_loss = nn.MSELoss().cuda()

        # keep parameters that need gradient
        params = [{'params': filter(lambda p: p.requires_grad, self.c3d.parameters()), 'lr': 0.1 * self.config.learning_rate},
                  {'params': self.vlatt_hier_lstm.parameters(), 'lr': self.config.learning_rate},
                  {'params': self.fc1.parameters(), 'lr': self.config.learning_rate},
                  {'params': self.fc2.parameters(), 'lr': self.config.learning_rate}]
        self.optimizer = optim.Adam(params, lr = self.config.learning_rate, weight_decay = self.config.weight_decay)

        # set model mode
        self.model.train()

    def train(self):
        step = 0
        for epoch_i in range(self.config.max_epoch_num):
            # one video, one batch
            for batch_i, filename in enumerate(self.train_set):
                # load data and gt
                filename += '.h5'
                h5 = h5py.File(filename, 'r')
                data = np.array(h5['data'])
                gt = np.array(h5['gt'])
                h5.close()

                # extract c3d features
                video_fea = None
                # for every clip (16 frames); prevent 'out of memory'
                for i in range(len(data)):
                    clip = np.expand_dims(data[i], axis = 0)  # 1, ch, d, h, w
                    clip = np.float32(clip)
                    # numpy array -> torch tensor -> torch variable -> torch gpu variable
                    clip_fea = self.c3d(Variable(torch.from_numpy(clip)).cuda())
                    # torch gpu variable -> torch gpu tensor -> torch tensor -> numpy array
                    clip_fea = clip_fea.data.cpu().numpy()
                    if video_fea is None:
                        video_fea = clip_fea
                    else:
                        video_fea = np.vstack((video_fea, clip_fea))

                # numpy array -> torch tensor -> torch variable -> torch gpu variable
                video_fea = Variable(torch.from_numpy(video_fea)).cuda()
                video_fea = video_fea.view(len(video_fea), 1, -1)

                # get video spatial-temporal representation
                video_rep = self.vlatt_hier_lstm(video_fea)

                # get scores
                video_rep = video_rep.view(-1, self.config.hidden_size)
                scores = F.relu(self.fc1(video_rep))
                scores = F.sigmoid(self.fc2(scores))

                # numpy array -> torch tensor -> torch variable -> torch gpu variable
                gt = Variable(torch.from_numpy(np.float32(gt))).cuda()

                # calculate loss
                loss = self.mse_loss(scores, gt.view(len(gt), -1))

                pdb.set_trace()

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step += 1

    def test(self):
        pass