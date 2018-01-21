import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import os
import sys
import h5py
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

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

        # linear compression
        self.lcp = nn.Linear(4096, self.config.input_size).cuda()

        # forward variant lstm for spatial-temporal representation
        self.forward_lstm = VLAttHierLstm(self.config.input_size, self.config.hidden_size,
                                             self.config.num_layers).cuda()

        # backward variant lstm for spatial-temporal representation
        self.backward_lstm = VLAttHierLstm(self.config.input_size, self.config.hidden_size,
                                           self.config.num_layers).cuda()


        # get scores
        self.fc_past = nn.Linear(self.config.hidden_size, self.config.hidden_size).cuda()
        self.fc_future = nn.Linear(self.config.hidden_size, self.config.hidden_size).cuda()
        self.fc_middle = nn.Linear(self.config.hidden_size, 128).cuda()
        self.fc_top = nn.Linear(128, 1).cuda()

        self.model = nn.ModuleList([self.c3d, self.lcp, self.forward_lstm, self.backward_lstm,
                                    self.fc_past, self.fc_future, self.fc_middle, self.fc_top])

        # minimal squared error loss
        self.mse_loss = nn.MSELoss().cuda()

        # keep parameters that need gradient
        params = [{'params': filter(lambda p: p.requires_grad, self.c3d.parameters()), 'lr': 0.1 * self.config.learning_rate},
                  {'params': list(self.lcp.parameters()) + list(self.forward_lstm.parameters()) +
                            list(self.backward_lstm.parameters()) + list(self.fc_past.parameters()) +
                            list(self.fc_future.parameters()) + list(self.fc_middle.parameters()) +
                            list(self.fc_top.parameters()),
                   'lr': self.config.learning_rate}]
        self.optimizer = optim.Adam(params, lr = self.config.learning_rate, weight_decay = self.config.weight_decay)

        # set model mode
        self.model.train()

    def train(self):
        step = 0
        minimal_loss = sys.float_info.max
        for epoch_i in trange(self.config.max_epoch_num, desc = 'Epoch', ncols = 60):
            # keep track of loss
            loss_history = []

            # one video, one batch
            for batch_i, filename in enumerate(tqdm(self.train_set, desc = 'Batch', ncols = 60, leave = False)):
                # load data and gt
                filename += '.h5'
                h5 = h5py.File(filename, 'r')
                video = np.array(h5['video'])
                gt = np.array(h5['gt'])
                h5.close()

                print(filename)

                duration = 16  # 16 frames for 3d cnn
                forward_video_fea = None    # video c3d features
                # extract video frames
                vidcap = cv2.VideoCapture(str(video))  # major version of cv >= 3
                num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                # delete the last several frames (< 16 frames)
                num_frames -= (num_frames % duration)
                cnt = 0
                clip = None
                while vidcap.isOpened() and cnt < num_frames:
                    success, image = vidcap.read()
                    if success:
                        # resize for c3d
                        image = cv2.resize(image, (112, 112))
                        if cnt % duration == 0:
                            clip = image.reshape((1, -1, 112, 112))
                        else:
                            clip = np.vstack((clip, image.reshape(1, -1, 112, 112)))
                        # for every clip (16 frames); prevent 'out of memory'
                        if (cnt + 1) % duration == 0:
                            clip = clip.transpose(1, 0, 2, 3)  # ch, d, h, w
                            clip = np.float32(np.expand_dims(clip, axis = 0))  # 1, ch, d, h, w
                            # numpy array -> torch tensor -> torch variable -> torch gpu variable
                            clip_fea = self.c3d(Variable(torch.from_numpy(clip)).cuda())
                            # compress video feature
                            clip_fea = self.lcp(clip_fea)
                            # concatenate clip features to video feature
                            if forward_video_fea is None:
                                forward_video_fea = clip_fea
                            else:
                                pdb.set_trace()
                                forward_video_fea = torch.cat((forward_video_fea, clip_fea))
                        cnt += 1
                    else:
                        break

                # free memory
                cv2.destroyAllWindows()
                vidcap.release()

                pdb.set_trace()

                # adjust dimensions
                forward_video_fea = torch.cat(forward_video_fea)
                forward_video_fea = forward_video_fea.view(len(forward_video_fea), 1, -1)

                # flip tensor along the first axis
                inv_idx = torch.arange(forward_video_fea.size(0) - 1, -1, -1).long()
                backward_video_fea = torch.index_select(forward_video_fea, 0, inv_idx)

                # get forward spatial-temporal representation
                forward_rep = self.forward_lstm(Variable(forward_video_fea).cuda())

                # get backward spatil-temporal representation
                backward_rep = self.backward_lstm(Variable(backward_video_fea).cuda())

                # flip tensor along the first axis, for ease of calculating scores
                inv_idx = torch.arange(backward_rep.size(0) - 1, -1, -1).long()
                backward_rep = torch.index_select(backward_rep, 0, Variable(inv_idx, requires_grad = False).cuda())

                pdb.set_trace()

            #     # get scores
            #     video_rep = video_rep.view(-1, self.config.hidden_size)
            #     scores = F.relu(self.fc1(video_rep))
            #     scores = F.sigmoid(self.fc2(scores))
            #
            #     # numpy array -> torch tensor
            #     gt = torch.from_numpy(np.float32(gt))
            #
            #     # calculate loss
            #     # torch tensor -> torch variable -> torch gpu variable
            #     batch_loss = self.mse_loss(scores, Variable(gt.view(len(gt), -1)).cuda())
            #
            #     # back propagation
            #     self.optimizer.zero_grad()
            #     batch_loss.backward()
            #     self.optimizer.step()
            #
            #     # record current loss
            #     loss_history.append(batch_loss.data.cpu())
            #
            #     step += 1
            #
            # # plot all batch loss in this epoch
            # self.plot_loss(loss_history)
            #
            # # average all batch loss in this epoch
            # epoch_loss = torch.stack(loss_history).mean()
            #
            # if epoch_loss < minimal_loss:
            #     minimal_loss = epoch_loss
            #     # save model
            #     videosm_model = os.path.join(str(self.config.videosm_model_dir), 'videosm_epoch_' + str(epoch_i + 1) + '.pkl')
            #     torch.save(self.model.state_dict(), videosm_model)
            # else:
            #     break

    def test(self):
        pass

    def plot_loss(self, losses):

        plt.plot(np.linspace(0, len(losses), num = len(losses)), losses)
        # axis ranges
        plt.axis([0, len(losses), 0, 1.2 * max(losses)])
        plt.title('Batch Loss')
        plt.xlabel('batch index')
        plt.ylabel('loss')
        plt.show()