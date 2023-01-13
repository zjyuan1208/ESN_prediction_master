# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/narma10_esn
# Description : NARMA-10 prediction with ESN.
# Date : 26th of January, 2018
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>


# Imports
import torch
from echotorch.datasets.TimeSeriesDataset import TimeSeriesDataset
import echotorch.nn.reservoir as etrs
import echotorch.utils
import echotorch.utils.matrix_generation as mg
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# eval state
eval_state = False
input_train = False
train_weight_state = True
shift = 200

num_layer = 2

# Length of training samples
# train_sample_length = 20000
train_sample_length = 200


# Length of test samples
test_sample_length = 20000

# How many training/test samples
n_train_samples = 90
n_test_samples = 1000

# Batch size (how many sample processed at the same time?)
batch_size = 100

# Reservoir hyper-parameters
spectral_radius = 0.6
leaky_rate = 0.1
input_dim = 2
output_dim = 2
reservoir_size = 100
connectivity = 0.1
ridge_param = 0.0000001

# Predicted/target plot length
plot_length = 30000

# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed initialisation
np.random.seed(1)
torch.manual_seed(1)

eps0 = 2e-2    # learning rate
Nep = 200
# Ns = 20000
Ns = 900 * 20000
N = 8
M = np.zeros((0, N))
Num = 1000      # size of the minibatch
mu = 0.0
sigma = 1.0
hid = 100
synapses = np.random.normal(mu, sigma, (hid, N))
prec = 1e-30
delta = 0.4    # Strength of the anti-hebbian learning
p = 2.0        # Lebesgue norm of the weights
k = 2          # ranking parameter, must be integer that is bigger or equal than 2

if input_train:
    time_train_dataset = TimeSeriesDataset(train_sample_length, n_train_samples, shift_time=shift, eval=False)
    trainloader = DataLoader(time_train_dataset, batch_size=900, shuffle=False)

    # For each batch
    for data in trainloader:
        # Inputs and outputs
        X, _ = data

        M = X.view(X.size(0)*X.size(1), -1).numpy()
        # targets = targets.numpy()

        # Transform data to Variables
        # inputs, targets = Variable(inputs), Variable(targets)
        # if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
    for nep in range(Nep):
        eps = eps0 * (1 - nep / Nep)
        M = M[np.random.permutation(Ns), :]
        for i in range(Ns // Num):
            inputs = np.transpose(M[i * Num:(i + 1) * Num, :])
            sig = np.sign(synapses)
            tot_input = np.dot(sig * np.absolute(synapses) ** (p - 1), inputs)

            y = np.argsort(tot_input, axis=0)
            yl = np.zeros((hid, Num))
            yl[y[hid - 1, :], np.arange(Num)] = 1.0
            yl[y[hid - k], np.arange(Num)] = -delta

            xx = np.sum(np.multiply(yl, tot_input), 1)
            ds = np.dot(yl, np.transpose(inputs)) - np.multiply(np.tile(xx.reshape(xx.shape[0], 1), (1, N)), synapses)

            nc = np.amax(np.absolute(ds))
            if nc < prec:
                nc = prec
            synapses += eps * np.true_divide(ds, nc)

# np.save('weight.npy', synapses)
# synapses = np.load('weight.npy')
# w_in = torch.tensor(synapses)


# Get matrix generators
matrix_generator = mg.matrix_factory.get_generator(
    name='normal',
    connectivity=0.1,
    mean=0.0,
    std=1.0
)

bias_matrix_generator = mg.matrix_factory.get_generator(
    name='normal',
    connectivity=0.1,
    mean=0.0,
    std=0.000001
)

# Create a Leaky-integrated ESN,
# with least-square training algo.
# esn = etrs.ESN(
esn = etrs.DeepESN(
    n_layers= num_layer,
    input_dim=input_dim,
    hidden_dim=reservoir_size,
    output_dim=output_dim,
    w_generator=matrix_generator,
    win_generator=matrix_generator,
    wbias_generator=bias_matrix_generator,
    leak_rate=leaky_rate,
    # spectral_radius=spectral_radius,
    learning_algo='pinv',
    input_scaling=1.0,
    input_type='IA',
    # bias_scaling=0,
    ridge_param=ridge_param,
    train_weight=train_weight_state
    # eval=False
)

# Transfer in the GPU if possible
if use_cuda:
    esn.cuda()
# end if

if eval_state is False:
    train_dataset = TimeSeriesDataset(train_sample_length, n_train_samples, shift_time=shift, input_dim=input_dim, output_dim=output_dim, extra_dim=None, eval=False)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # For each batch
    for data in trainloader:
        # Inputs and outputs
        inputs, targets = data

        # Transform data to Variables
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        # ESN need inputs and targets
        esn(inputs, targets)
    # end for

    # Now we finalize the training by computing the output matrix Wout.
    esn.finalize()

    # Get the first sample in training set, and transform it to Variable.
    dataiter = iter(trainloader)
    train_u, train_y = dataiter.next()
    train_u, train_y = Variable(train_u), Variable(train_y)
    if use_cuda: train_u, train_y = train_u.cuda(), train_y.cuda()

    # Make a prediction with our trained ESN
    y_predicted = esn(train_u)

    # Print training MSE and NRMSE
    print(u"Train MSE: {}".format(echotorch.utils.mse(y_predicted.data, train_y.data)))
    print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, train_y.data)))
    print(u"")
    pt_path = r'C:\Users\zhyuan\OneDrive - UGent\Desktop\ESN\checkpoint\2layer_esn_2in2out_0.05_0.2_IA.pt'

    torch.save(esn, pt_path)
else:
    test_dataset = TimeSeriesDataset(test_sample_length, n_test_samples, shift_time=shift, input_dim=input_dim, output_dim=output_dim, extra_dim=None, eval=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataiter = iter(testloader)
    test_u, test_y = dataiter.next()
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()

    # Make a prediction with our trained ESN
    new_m = torch.load(r'C:\Users\zhyuan\OneDrive - UGent\Desktop\ESN\checkpoint\2layer_esn_2in2out_0.05_0.2_IA.pt')
    y_predicted = new_m(test_u)


    # y_predicted = esn(test_u)

    # Print test MSE and NRMSE
    print(u"Test MSE: {}".format(echotorch.utils.mse(y_predicted.data, test_y.data)))
    print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, test_y.data)))
    print(u"")

    # for i in range(8):
    #     plt.plot(test_y[4, :plot_length, i].data, 'r')
    #     plt.plot(y_predicted[4, :plot_length, i].data, 'b')
    #     plt.show()
    plt.plot(test_y[3, :plot_length, 0].data, 'r', label='feat1_ori')
    plt.plot(y_predicted[3, :plot_length, 0].data, 'b', label='feat1_pred')
    plt.legend()
    plt.title('2_in_2_out_1')
    plt.show()

    plt.plot(test_y[3, :plot_length, 1].data, 'y', label='feat2_ori')
    plt.plot(y_predicted[3, :plot_length, 1].data, 'k', label='feat2_pred')
    plt.legend()
    plt.title('2_in_2_out_2')
    plt.show()

    plt.plot(test_y[5, :plot_length, 0].data, 'r', label='feat1_ori')
    plt.plot(y_predicted[5, :plot_length, 0].data, 'b', label='feat1_pred')
    plt.legend()
    plt.title('2_in_2_out_1')
    plt.show()

    plt.plot(test_y[5, :plot_length, 1].data, 'y', label='feat2_ori')
    plt.plot(y_predicted[5, :plot_length, 1].data, 'k', label='feat2_pred')
    plt.legend()
    plt.title('2_in_2_out_2')
    plt.show()

    plt.plot(test_y[14, :plot_length, 0].data, 'r', label='feat1_ori')
    plt.plot(y_predicted[14, :plot_length, 0].data, 'b', label='feat1_pred')
    plt.legend()
    plt.title('2_in_2_out_1')
    plt.show()

    plt.plot(test_y[14, :plot_length, 1].data, 'y', label='feat2_ori')
    plt.plot(y_predicted[14, :plot_length, 1].data, 'k', label='feat2_pred')
    plt.legend()
    plt.title('2_in_2_out_2')
    plt.show()


    # plt.plot(test_y[6, :plot_length, 0].data, 'r')
    # plt.plot(y_predicted[6, :plot_length, 0].data, 'b')
    # # plt.show()
    #
    # plt.plot(test_y[6, :plot_length, 1].data, 'y')
    # plt.plot(y_predicted[6, :plot_length, 1].data, 'k')
    # plt.show()
    #
    # plt.plot(test_y[7, :plot_length, 0].data, 'r')
    # plt.plot(y_predicted[7, :plot_length, 0].data, 'b')
    # # plt.show()
    #
    # plt.plot(test_y[7, :plot_length, 1].data, 'y')
    # plt.plot(y_predicted[7, :plot_length, 1].data, 'k')
    # plt.show()

