# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
# Imports
import torch
from torch.utils.data.dataset import Dataset
import scipy.io
import matplotlib
import mat73


# 10th order NARMA task
class TimeSeriesDataset(Dataset):
    """
    xth order NARMA task
    WARNING: this is an unstable dataset. There is a small chance the system becomes
    unstable, leading to an unusable dataset. It is better to use NARMA30 which
    where this problem happens less often.
    """

    # Constructor
    def __init__(self, sample_len, n_samples, shift_time, input_dim, output_dim, extra_dim, eval=False):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param system_order: th order NARMA
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        # self.system_order = system_order
        self.shift_time = shift_time
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.extra_dim = extra_dim

        # System order
        # self.parameters = torch.zeros(4)
        # # if system_order == 10:
        # #     self.parameters[0] = 0.3
        # #     self.parameters[1] = 0.05
        # #     self.parameters[2] = 9
        # #     self.parameters[3] = 0.1
        # # else:
        # self.parameters[0] = 0.2
        # self.parameters[1] = 0.04
        # self.parameters[2] = 29
        # self.parameters[3] = 0.001
        # end if

        # Generate data set
        self.inputs, self.outputs = self._generate(eval)
    # end __init__

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return self.inputs[idx], self.outputs[idx]
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Generate
    def _generate(self, eval=False):
        """
        Generate dataset
        :return:
        """
        # inputs = list()
        # outputs = list()
        # for i in range(self.n_samples):
        #     ins = torch.rand(self.sample_len, 2) * 0.5
        #     outs = torch.zeros(self.sample_len, 2)
        #     for k in range(self.system_order - 1, self.sample_len - 1):
        #         outs[k+1] = ins[k]
        #     inputs.append(ins)
        #     outputs.append(outs)

        # data = scipy.io.loadmat(r'C:\Users\zhyuan\OneDrive - UGent\Desktop\ESN\data\rhythm_train2.mat', )
        data = mat73.loadmat(r'C:\Users\zhyuan\OneDrive - UGent\Desktop\ESN\data\rhythm_train2_10000.mat')

        # data = scipy.io.loadmat(r'C:\Users\zhyuan\OneDrive - UGent\Desktop\ESN_prediction\examples\datasets\rhythm_train_1.mat')
        # Xtr = torch.FloatTensor(data['series']).transpose(-1, -2)[:, :, 0:2]
        Xtr = torch.FloatTensor(data['series']).transpose(-1, -2)

        inputs = list()
        outputs = list()

        if eval is False:
            for i in range(self.n_samples):
                # ins = Xtr[i][0:self.sample_len:, :]
                ins = Xtr[:, :, 0:self.input_dim][i][0:self.sample_len, :]
                if self.extra_dim is not None:
                    ins_target = Xtr[:, :, 2:][i][0:self.sample_len, :]
                # outs = torch.zeros(self.sample_len, 6)
                outs = torch.zeros(self.sample_len, self.output_dim)

                for k in range(0, self.sample_len - self.shift_time):
                    if self.extra_dim is not None:
                        outs[k] = ins_target[k + self.shift_time]
                    else:
                        outs[k] = ins[k + self.shift_time]
                inputs.append(ins)
                outputs.append(outs)
        else:
            for i in range(899):
                ins = Xtr[:, :, 0:self.input_dim][i][0:self.sample_len, :]
                if self.extra_dim is not None:
                    ins_target = Xtr[:, :, 2:][i][0:self.sample_len, :]
                # outs = torch.zeros(self.sample_len, 6)
                outs = torch.zeros(self.sample_len, self.output_dim)

                for k in range(0, self.sample_len - self.shift_time):
                    if self.extra_dim is not None:
                        outs[k] = ins_target[k + self.shift_time]
                    else:
                        outs[k] = ins[k + self.shift_time]
                inputs.append(ins)
                outputs.append(outs)
                # # ins = Xtr[i+899][0:self.sample_len:, :]
                # ins = Xtr[:, :, 0:self.input_dim][i+899][0:self.sample_len, :]
                # # ins_target = Xtr[:, :, self.output_dim:][i][0:self.sample_len, :]
                # # outs = torch.zeros(self.sample_len, 6)
                # outs = torch.zeros(self.sample_len, self.output_dim)
                #
                # for k in range(0, self.sample_len - self.shift_time):
                #     # outs[k] = ins_target[k + self.shift_time]
                #     outs[k] = ins[k + self.shift_time]
                # inputs.append(ins)
                # outputs.append(outs)

        # inputs = list()
        # outputs = list()
        #
        # dt = 1.0
        # T = 60 * int(200/dt)
        # V = 2
        # N = self.n_samples
        #
        # BPM = torch.tensor([95, 120])
        # interval = torch.tensor([int(BPM[0]/60*1000/dt), int(BPM[1]/60*1000/dt)])
        # multiples = torch.tensor([1, 2, 3])
        #
        # t0 = int(2000/dt)
        # sigma = 300/dt
        # t_basis = torch.linspace(start=1, end=int(1000/dt), steps=1)
        # basis = t_basis * torch.exp(-t_basis ** 2/sigma**2)
        # basis = basis.reshape([1, 1, len(t_basis)])
        # series = torch.zeros([N, V, T])
        # outs = torch.zeros([T, V])
        #
        #
        # for i in range(self.n_samples):
        #     beat = interval[0]+torch.rand([1])*(interval[1]-interval[0])
        #     a = torch.rand([V, 1]).round() * (len(multiples) - 1)
        #     ind_list = a.view(1, -1).numpy().tolist()
        #     multiple = multiples[ind_list]
        #     total_interval = torch.tensor(beat*multiple).round()
        #     for channel in range(V):
        #         lst = []
        #         for j in range(t0, T, int(total_interval[channel])):
        #             lst.append(j)
        #         hit_locations = torch.tensor(lst)
        #         for hit_i in range(len(hit_locations)):
        #             hit = hit_locations[hit_i]
        #             if hit + len(basis) - 1 < T:
        #                 series[i, channel, hit:hit+len(basis)-1] = series[i, channel, hit:hit+len(basis)-1] + basis
        # inputs = series.transpose(-1, -2).numpy().tolist()
        # inputs_lst = []
        # for i in range(len(inputs)):
        #     inputs_lst.append(torch.tensor(inputs[i]))
        #     for k in range(self.system_order - 1, self.sample_len - 1):
        #         outs[k] = torch.tensor(inputs[i])[k-50]
        #     outputs.append(outs)

        return inputs, outputs

# end NARMADataset
