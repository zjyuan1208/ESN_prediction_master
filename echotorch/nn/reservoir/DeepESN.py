import os

import torch
import torch.nn as nn
import numpy as np
import echotorch.utils.matrix_generation as mg
from echotorch.nn.linear.RRCell import RRCell
from echotorch.nn.linear.IncRRCell import IncRRCell
from ..Node import Node
from .LiESNCell import LiESNCell
import torch.optim as optim



# Deep ESN
class DeepESN(Node):
    """
    Deep ESN as defined in Gallicchio, C., Micheli, A., & Pedrelli, L. (2017).
    """

    # Constructor
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, w_generator, win_generator, wbias_generator,
                 leak_rate, input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, train_weight=False,
                 with_bias=True, softmax_output=False, normalize_output=False, washout=0, create_rnn=True,
                 create_output=True, input_type='IA', output_type='AO', niter=100, debug=Node.NO_DEBUG, test_case=None,
                 dtype=torch.float32):
        """
        Constructor
        :param n_layers: Number of layers to create
        :param input_dim: Input dimension size
        :param hidden_dim: Size of the reservoirs (all layer have the same size)
        :param output_dim: Output dimension size
        :param w_generator: Generator for the reservoir-to-reservoir matrices
        :param win_generator: Generator for the input-to-reservoir matrices (inputs for first layer, inputs for other from previous layer).
        :param wbias_generator: Generator for the internal biaises
        :param input_scaling: Input scaling (first layer)
        :param nonlin_func: Activation function (all layers)
        :param learning_algo: Learning algorithm (output layer) as 'inv' or 'pinv'
        :param ridge_param: Regularization parameter (output layer)
        :param with_bias: Add a bias to the output layer
        :param softmax_output: Add a softmax layer after the output layer ?
        :param washout: Washout period (ignore timesteps at the beginning of each sample)
        :param create_rnn: Create the RNN layers ?
        :param create_output: Create the output layer ?
        :param input_type: Input variant (IF: input-to-first, IA: input-to-all, GE: grouped-ESNs)
        :param output_type: Output flavour (AO: all-to-outputs, LO: last-to-outputs)
        :param debug: Debug mode
        :param test_case: Test case to call for test
        :param dtype: Data type
        """
        super(DeepESN, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            debug=debug,
            test_case=test_case,
            dtype=dtype
        )

        # Properties
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._with_bias = with_bias
        self._w_generator = w_generator
        self._win_generator = win_generator
        self._wbias_generator = wbias_generator
        self._input_type = input_type
        self._washout = washout
        self._dtype = dtype
        self._input_scaling = input_scaling
        self._nonlin_func = nonlin_func
        self.train_weight = train_weight

        # List of reservoirs
        self._reservoirs = list()

        # For SOM
        self.pdist = nn.PairwiseDistance(p=2)
        # self.m = self._hidden_dim
        self.m = 1
        self.n = self._hidden_dim
        # self.n = hidden_dim
        self.dim = hidden_dim
        self.weights = torch.randn(1, self._hidden_dim)
        # self.weights_ = torch.randn(self._hidden_dim*self._hidden_dim, self._hidden_dim)
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        # if alpha is None:
        self.alpha = 0.3
        # else:
        #     self.alpha = float(alpha)
        # if sigma is None:
        self.sigma = max(self.m, self.n) / 2.0
        # else:
        #     self.sigma = float(sigma)
        self.n_iter = niter

        # For output
        self.output_learning_rate = 0.1
        self.momentum = 0.2



        # Create each layer
        if create_rnn:
            leak_rate_lst = [0.05, 0.2]
            for layer_i in range(self._n_layers):
                # Generate matrices
                w, w_in, w_bias = self._generate_matrices(
                    self._get_hyperparam_value(w_generator, layer_i),
                    self._get_hyperparam_value(win_generator, layer_i),
                    self._get_hyperparam_value(wbias_generator, layer_i),
                    layer_i == 0
                )

                if self.train_weight == True:
                    if layer_i > 0:
                        w_in_layer1 = self._reservoirs[0].w_in
                        mask = self.mask(w_in_layer1.t())
                        synapses = torch.mul(self.weights, mask.t())

                        # synapses = np.load('inter_weight.npy')
                        w_in_cat = torch.rand([hidden_dim, input_dim])
                        w_in = synapses.t()
                        # w_in = torch.FloatTensor(synapses)
                        w_in = torch.cat((w_in, w_in_cat), 1)


                # Input dim
                layer_input_dim = w_in.size(1)

                # Recurrent layer
                esn_cell = LiESNCell(
                    input_dim=layer_input_dim,
                    output_dim=hidden_dim,
                    w=w,
                    w_in=w_in,
                    w_bias=w_bias,
                    leaky_rate=self._get_hyperparam_value(leak_rate_lst[layer_i], layer_i),
                    input_scaling=self._get_hyperparam_value(input_scaling, layer_i),
                    nonlin_func=self._get_hyperparam_value(nonlin_func, layer_i),
                    washout=washout,
                    debug=debug,
                    test_case=test_case,
                    dtype=dtype
                )

                # Add
                self._reservoirs.append(esn_cell)
            # end for
        # end if

        # Output layer
        if create_output:
            # self._output = RRCell(
            #     input_dim=hidden_dim * n_layers,
            #     output_dim=output_dim,
            #     ridge_param=ridge_param,
            #     with_bias=with_bias,
            #     learning_algo=learning_algo,
            #     softmax_output=softmax_output,
            #     normalize_output=normalize_output,
            #     debug=debug,
            #     test_case=test_case,
            #     dtype=dtype
            # )

            # self._output = IncRRCell(
            #     input_dim=hidden_dim,
            #     output_dim=output_dim,
            #     conceptors=None,
            #     ridge_param=ridge_param,
            #     with_bias=with_bias,
            #     learning_algo=learning_algo,
            #     softmax_output=softmax_output,
            #     debug=debug,
            #     test_case=test_case,
            #     dtype=dtype
            # )

            self._output = nn.Linear(in_features=hidden_dim*self._n_layers, out_features=output_dim)
            # self._output = linear(input)

            self.add_trainable(self._output)
        # end if
    # end __init__


    # Append a layer
    def append_layer(self, esn_cell):
        """
        Append a layer
        :param esn_cell: The ESNCell object to append to the stack
        """
        self._reservoirs.append(esn_cell)
        self._n_layers += 1
    # end append_layer

    # region PRIVATE

    # Get hyperparameter value
    def _get_hyperparam_value(self, hyperparam, layer_i):
        """
        Get hyperparameter value
        :param hyperparam: Hyperparameter (a value or a list)
        :param layer_i: Which layer (integer)
        :return: Hyperparameter value for this layer
        """
        if type(hyperparam) == list or type(hyperparam) == np.ndarray or type(hyperparam) == torch.tensor:
            return hyperparam[layer_i]
        else:
            return hyperparam
        # end if
    # end _get_hyperparam_value

    # Generate matrices
    def _generate_matrices(self, w_generator, win_generator, wbias_generator, first_layer=False):
        """
        Generate matrices
        :param w_generator: W matrix generator
        :param win_generator: Win matrix generator
        :param wbias_generator: Wbias matrix generator
        :return: W, Win, Wbias
        """
        # Generate W matrix
        if isinstance(w_generator, mg.MatrixGenerator):
            w = w_generator.generate(size=(self._hidden_dim, self._hidden_dim), dtype=self._dtype)
        elif callable(w_generator):
            w = w_generator(size=(self._hidden_dim, self._hidden_dim), dtype=self._dtype)
        else:
            w = w_generator
        # end if

        # Input matrix size
        if first_layer:
            win_size = (self._hidden_dim, self._input_dim)
        else:
            if self._input_type == 'IF':
                win_size = (self._hidden_dim, self._hidden_dim)
            elif self._input_type == 'IA':
                win_size = (self._hidden_dim, self._hidden_dim + self._input_dim)
            elif self._input_type == 'GE':
                win_size = (self._hidden_dim, self._input_dim)
            else:
                raise Exception("Unknown value for input_type : {}".format(self._input_type))
            # end if
        # end if

        # Generate Win matrix
        if isinstance(win_generator, mg.MatrixGenerator):
            w_in = win_generator.generate(size=win_size, dtype=self._dtype)
        elif callable(win_generator):
            w_in = win_generator(size=win_size, dtype=self._dtype)
        else:
            w_in = win_generator
        # end if

        # Generate Wbias matrix
        if isinstance(wbias_generator, mg.MatrixGenerator):
            w_bias = wbias_generator.generate(size=self._hidden_dim, dtype=self._dtype)
        elif callable(wbias_generator):
            w_bias = wbias_generator(size=self._hidden_dim, dtype=self._dtype)
        else:
            w_bias = wbias_generator
        # end if

        return w, w_in, w_bias
    # end _generate_matrices

    # endregion PRIVATE

    # region OVERRIDE

    def mask(self, w_in_layer1):
        tmp_mask = torch.zeros_like(w_in_layer1)
        for i in range(w_in_layer1.size(0)):
            ind = torch.nonzero(w_in_layer1[i], as_tuple=False)
            for j in range(ind.size(0)):
                tmp_mask[i][ind[j]] = i + 1
        if tmp_mask.size(0) > 2:
            res = torch.eq(tmp_mask[0], tmp_mask[1])
            # res_final = torch.zeros(res.size())
            for k in range(tmp_mask.size(0) - 2):
                res = torch.eq(res, tmp_mask[2 + k])
        else:
            res = torch.eq(tmp_mask[0], tmp_mask[1])
        res_final = torch.zeros(res.size())
        for i in range(res.size(0)):
            if res[i] == False:
                res_final[i] = 0.0
            else:
                res_final[i] = 1.0
        mask = res_final
        return mask

    def hebbian_train_layer(self, input):
        eps0 = 2e-2  # learning rate
        Nep = 200
        # Ns = 20000
        # Ns = 100 * 20000
        N = 100
        M = np.zeros((0, N))
        Num = 1000  # size of the minibatch
        mu = 0.0
        sigma = 1.0
        hid = 100
        synapses = np.random.normal(mu, sigma, (hid, N))
        prec = 1e-30
        delta = 0.4  # Strength of the anti-hebbian learning
        p = 2.0  # Lebesgue norm of the weights
        k = 2  # ranking parameter, must be integer that is bigger or equal than 2

        X = input
        M = X.view(X.size(0) * X.size(1), -1).numpy()
        Ns = X.size(0) * X.size(1)

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
                ds = np.dot(yl, np.transpose(inputs)) - np.multiply(np.tile(xx.reshape(xx.shape[0], 1), (1, N)),
                                                                    synapses)

                nc = np.amax(np.absolute(ds))
                if nc < prec:
                    nc = prec
                synapses += eps * np.true_divide(ds, nc)

        return synapses

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def som_train_layer(self, x, it):
        dists = self.pdist(torch.stack([x for i in range(self.m * self.n)]), self.weights)
        # dists = self.pdist(torch.stack([x for i in range(self._hidden_dim)]), self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index, :]
        bmu_loc = bmu_loc.squeeze()

        learning_rate_op = 1.0 - it / self.n_iter
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        bmu_distance_squares = torch.sum(
            torch.pow(self.locations.float() - torch.stack([bmu_loc for i in range(self.m * self.n)]).float(), 2), 1)

        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op ** 2)))

        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack(
            [learning_rate_op[i:i + 1].repeat(self.dim) for i in range(self.m * self.n)])
            # [learning_rate_op[i:i + 1].repeat(self.dim) for i in range(self._hidden_dim)])
        delta = torch.mul(learning_rate_multiplier, (torch.stack([x for i in range(self.m * self.n)]) - self.weights))
        # delta = torch.mul(learning_rate_multiplier, (torch.stack([x for i in range(self._hidden_dim)]) - self.weights))
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights

        return self.weights, self.locations

    def get_weights(self):
        return self.weights_

    def get_locations(self):
        return self.locations

    # Forward
    def forward(self, u, y=None, reset_state=True):
        """
        Forward function
        :param u: Input signal
        :param y: Target outputs (or None if prediction)
        :param reset_state: Reset hidden state to zero or keep old one ?
        :return: Output (eval) or hidden states (train)
        """
        # Sizes
        time_length = int(u.size(1))
        batch_sizes = int(u.size(0))

        # Keep hidden states
        # hidden_states = torch.zeros(batch_sizes, time_length, self._hidden_dim * self._n_layers, dtype=self._dtype)
        hidden_states = torch.zeros(batch_sizes, time_length, self._hidden_dim * self._n_layers, dtype=self._dtype)

        # Input to first layer
        layer_input = u

        # Compute hidden states for each layer
        for layer_i in range(self._n_layers):
            # Feed ESN
            if self._input_type == 'IF':
                layer_hidden_states = self._reservoirs[layer_i](layer_input, reset_state=reset_state)
                hidden_states[:, :, layer_i*self._hidden_dim:(layer_i+1)*self._hidden_dim] = layer_hidden_states
                layer_input = layer_hidden_states
            elif self._input_type == 'IA':
                # Add inputs for upper layers
                if layer_i > 0:
                    layer_input_with_u = torch.cat((layer_input, u), dim=2)
                else:
                    layer_input_with_u = u
                # end if

                # Go through ESN cell
                # Replace the weight of the interlayer
                if layer_i == 0:
                    layer_hidden_states = self._reservoirs[layer_i](layer_input_with_u, reset_state=reset_state)
                    hidden_states[:, :, layer_i * self._hidden_dim:(layer_i + 1) * self._hidden_dim] = layer_hidden_states
                    layer_input = layer_hidden_states  # [50, 20000, 200]
                else:
                    # layer_hidden_states = self._reservoirs[layer_i](layer_input_with_u, reset_state=reset_state)
                    # hidden_states[:, :, layer_i * self._hidden_dim:(layer_i + 1) * self._hidden_dim] = layer_hidden_states
                    # layer_input = layer_hidden_states  # [50, 20000, 200]
                    if self.train_weight is True:
                        w, w_in, w_bias = self._generate_matrices(
                            self._get_hyperparam_value(self._w_generator, layer_i),
                            self._get_hyperparam_value(self._win_generator, layer_i),
                            self._get_hyperparam_value(self._wbias_generator, layer_i),
                            layer_i == 0
                        )

                        # Apply mask on the input to the second layer
                        w_in_layer1 = self._reservoirs[0].w_in
                        mask = self.mask(w_in_layer1.t())
                        # layer_input = torch.mul(layer_input, mask.t())
                        mask = torch.tensor(mask)
                        index = torch.nonzero(mask, as_tuple=False)
                        b = torch.zeros([layer_input.size(0), len(index), layer_input.size(-1)])
                        for num_batch in range(layer_input.size(0)):
                            for i in range(len(index)):
                                b[num_batch][i] = layer_input[num_batch][index[i]]

                        layer_input = b

                        # Train the weight between 2 reservoirs
                        # w_in = self.hebbian_train_layer(layer_hidden_states)
                        for iter_no in range(self.n_iter):
                            for i in range(len(layer_input)):
                                for j in range(len(layer_input[i])):
                                    self.som_train_layer(layer_input[i][j], iter_no)

                        # centroid_grid = [[] for i in range(self.m)]
                        # weights = self.get_weights()
                        # locations = self.get_locations()
                        # for i, loc in enumerate(locations):
                        #     centroid_grid[loc[0]].append(weights[i].numpy())
                        # w_in = self.som_train_layer(layer_input, it=100)
                        # np.save('weight_in.npy', self.weights)
                        # synapses = np.load('weight_in.npy')
                        # print('Weight saved!')
                        # w_in = torch.tensor(synapses)

                        # w_in_layer1 = self._reservoirs[0].w_in
                        # mask = self.mask(w_in_layer1.t())
                        # synapses = torch.mul(self.weights, mask.t())
                        #
                        w_in_cat = torch.rand([self._hidden_dim, self.input_dim])
                        # w_in = synapses
                        w_in = self.weights
                        w_in = torch.cat((w_in, w_in_cat), 1)

                        layer_input_dim = w_in.size(1)


                        esn_cell_trained = LiESNCell(
                            input_dim=layer_input_dim,
                            output_dim=self._hidden_dim,
                            w=w,
                            w_in=w_in,
                            w_bias=w_bias,
                            # leaky_rate=self._get_hyperparam_value(self.leak_rate_lst[layer_i], layer_i),
                            leaky_rate=self._get_hyperparam_value(0.2, layer_i),
                            input_scaling=self._get_hyperparam_value(self._input_scaling, layer_i),
                            nonlin_func=self._get_hyperparam_value(self._nonlin_func, layer_i),
                            # washout=washout,
                            # debug=self.debug,
                            # test_case=test_case,
                            dtype=self.dtype
                        )

                        self._reservoirs[layer_i] = esn_cell_trained
                    layer_hidden_states = self._reservoirs[layer_i](layer_input_with_u, reset_state=reset_state)
                    hidden_states[:, :, layer_i * self._hidden_dim:(layer_i + 1) * self._hidden_dim] = layer_hidden_states
                    layer_input = layer_hidden_states  # [batch_size, time_step, hidden_size]


            elif self._input_type == 'GE':
                layer_hidden_states = self._reservoirs[layer_i](u, reset_state=reset_state)
                hidden_states[:, :, layer_i * self._hidden_dim:(layer_i + 1) * self._hidden_dim] = layer_hidden_states
            else:
                raise Exception("Unknown input type : {}".format(self._input_type))
            # end if
        # end for

        # Learning algo
        if not self.training:
            # return self._output(hidden_states, None)
            return self._output(hidden_states)
        else:
            optimizer = optim.SGD(self._output.parameters(), lr=self.output_learning_rate, momentum=self.momentum)
            criterion = nn.MSELoss()
            out = self._output(hidden_states)
            loss = criterion(out, y)
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()
            return out
            # return self._output(hidden_states, y[:, self._washout:])
        # end if
    # end forward

    # Reset layer (not trained)
    def reset(self):
        """
        Reset layer (not trained)
        """
        # Reset output layer
        self._output.reset()

        # Training mode again
        self.train(True)
    # end reset

    # Reset hidden layer
    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        for layer_i in range(self._n_layers):
            self._reservoirs[layer_i].reset_hidden()
        # end for
    # end reset_hidden

    # Get item (get layer)
    def __getitem__(self, item):
        """
        Get item (get layer)
        :param item: Item index
        :return: ESNCell at item-th layer
        """
        return self._reservoirs[item]
    # end __getitem__

    # Set item (set layer)
    def __setitem__(self, key, value):
        """
        Set item (set layer)
        :param key: Layer index
        :param value: ESNCell object
        """
        self._reservoirs[key] = value
    # end __setitem__

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        :return: String
        """
        s = super(DeepESN, self).extra_repr()
        s += ', layers=[\n'
        for layer_i in range(self._n_layers):
            s += '\t{_reservoirs[' + str(layer_i) + ']},\n'
        # end for
        s += ']'
        return s.format(**self.__dict__)
    # end extra_repr

    # endregion OVERRIDE

# end DeepESN