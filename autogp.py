import copy
import torch
import warnings
import gpytorch
import pandas as pd
import torch.nn as nn
from math import floor
from deepgp import AutoDKL

class AutoGP():
    def __init__(self, name, hyper_params):
        super(AutoGP, self).__init__()
        self.name = name # task_id
        self.hyper_params = hyper_params # hyperparameters
        self._deep_template = self._get_network_instance() # Autoregressive deep kernel learning
        self._model_params = self._deep_template.state_dict() # model parameters

    def data_loader(self):
        #load data
        df_data = pd.read_csv(self.hyper_params["file_path"], sep=' ')[self.hyper_params["cols"][:self.hyper_params["num_nodes"]]]

        # data normalization
        train_data = df_data[:self.hyper_params["train_split"]]
        self.mean = train_data.mean(axis=0) # mean of dataframe
        self.std = train_data.std(axis=0) # standard deviation of dataframe
        data = (df_data - self.mean)/self.std
        data = torch.Tensor(data.values)

        # split the data
        unfold_data = data.unfold(0, self.hyper_params["seq_len"]+1, 1).transpose(1,2)
        train = unfold_data[:self.hyper_params["train_split"]-self.hyper_params["pre_horizon"]][::self.hyper_params["moving_step"]]
        test = unfold_data[self.hyper_params["train_split"]-self.hyper_params["pre_horizon"]:self.hyper_params["test_split"]-self.hyper_params["pre_horizon"]]
        val = unfold_data[self.hyper_params["test_split"]-self.hyper_params["pre_horizon"]:]
        train_x = train[:,:self.hyper_params["seq_len"],]
        train_y = train[:,self.hyper_params["seq_len"]:,].squeeze(-2)
        val_x = val[:,:self.hyper_params["seq_len"]]
        val_y = val[:,self.hyper_params["seq_len"]:,].squeeze(-2)

        # build a task matrix on training set
        full_train_i = torch.full((train_x.shape[0],1), dtype=torch.long, fill_value=0)
        for i in range(1,train_x.shape[-1]):
            train_i_task = torch.full((train_x.shape[0],1), dtype=torch.long, fill_value=i)
            full_train_i = torch.cat([full_train_i, train_i_task], 1)

        # build a task matrix on validation set
        full_val_i = torch.full((val_x.shape[0],1), dtype=torch.long, fill_value=0)
        for j in range(1,val_x.shape[-1]):
            val_i_task = torch.full((val_x.shape[0],1), dtype=torch.long, fill_value=j)
            full_val_i = torch.cat([full_val_i, val_i_task], 1)

        # build a task matrix on test set
        full_test_i = torch.full((test.shape[0]-self.hyper_params["pre_horizon"],1), dtype=torch.long, fill_value=0)
        for j in range(1,self.hyper_params["num_nodes"]):
            test_i_task = torch.full((test.shape[0]-self.hyper_params["pre_horizon"],1), dtype=torch.long, fill_value=j)
            full_test_i = torch.cat([full_test_i, test_i_task], 1)


        # move dataset and task matrix into GPUs
        if torch.cuda.is_available():
            train_x, train_y, val_x, val_y, test = train_x.cuda(), train_y.cuda(), val_x.cuda(), val_y.cuda(), test.cuda()
            full_train_i, full_val_i, full_test_i = full_train_i.reshape(-1).cuda(), full_val_i.reshape(-1).cuda(), full_test_i.reshape(-1).cuda()
        return train_x, train_y, val_x, val_y, test, full_train_i, full_val_i, full_test_i

    def kernel_initialization(self):
        # generate initate kenrel state
        final_kernel = 0
        for i in range(self.hyper_params["R"]):
            #this_iter = base_kernel_set.copy()
            add_kernel = copy.deepcopy(self.hyper_params["base_kernel_set"][0])
            for j in range(1,len(self.hyper_params["base_kernel_set"])):
                add_kernel += copy.deepcopy(self.hyper_params["base_kernel_set"][j])

            if i == 0:
                mul_kernel = copy.deepcopy(add_kernel)
                final_kernel = copy.deepcopy(mul_kernel)
            else:
                mul_kernel *= copy.deepcopy(add_kernel)
                final_kernel += gpytorch.kernels.ScaleKernel(copy.deepcopy(mul_kernel))
        return final_kernel

    def kernel_selection(self):
        # select suitable kernels
        self._deep_template.load_state_dict(self._model_params)

        constrains = []
        for constraint_name, constraint in self._deep_template.named_constraints():
            constrains.append(constraint)

        index = 0
        d = len(self.hyper_params["base_kernel_set"])
        a1 = len(self.hyper_params["base_kernel_set"])
        an = a1+(self.hyper_params["R"]-1)*d
        length = int((a1+an)*self.hyper_params["R"]/2)
        kernel_weights = torch.zeros(length)
        for param_name, param in self._deep_template.named_parameters():
            if 'covar_module' and 'raw_outputscale' in param_name:
                param_name = param_name.split('.')
                constrain = constrains[index]
                kernel_weights[index] = constraint.transform(param)
                index += 1

        pre = 0
        now = a1
        best_kernel_index = torch.zeros((self.hyper_params["R"],self.hyper_params["numbers"])).int()
        for r in range(self.hyper_params["R"]):
            weights = kernel_weights[pre:pre+d]
            pre = now
            now = a1 + a1+(r+1)*d
            weights = nn.functional.softmax(weights, dim=0)
            _, kernel_index = torch.sort(weights, descending=True)
            best_kernel_index[r] = kernel_index[:self.hyper_params["numbers"]]

        final_kernel = 0
        for i in range(self.hyper_params["R"]):
            add_kernel = copy.deepcopy(self.hyper_params["base_kernel_set"][best_kernel_index[i][0]])
            for j in range(1, self.hyper_params["numbers"]):
                add_kernel += copy.deepcopy(self.hyper_params["base_kernel_set"][best_kernel_index[i][j]])

            if i == 0:
                mul_kernel = copy.deepcopy(add_kernel)
                final_kernel = copy.deepcopy(mul_kernel)
            else:
                mul_kernel *= copy.deepcopy(add_kernel)
                final_kernel += gpytorch.kernels.ScaleKernel(copy.deepcopy(mul_kernel))
            return final_kernel

    def _get_network_instance(self):
	    # Gaussian procedure instantiation requires data to be loaded first
        self.train_x, self.train_y, self.val_x, self.val_y, self.test, self.full_train_i, self.full_val_i, self.full_test_i = self.data_loader()
        self.kernel = self.kernel_initialization()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood() # the likelihood function
        return AutoDKL((self.train_x, self.full_train_i), self.train_y.reshape(-1), self.likelihood, self.hyper_params, self.kernel)

    def fit(self):
        warnings.filterwarnings('ignore')
        if torch.cuda.is_available():
            self.likelihood= self.likelihood.cuda()
            self._deep_template = self._deep_template.cuda()

        optimizer = torch.optim.Adam(self._deep_template.parameters(), lr=self.hyper_params["learning_rate"]) # set optimizer
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self._deep_template) # set loss function

        counter = 0 # record the account of iterations that fail to decrease the validation loss
        min_loss = self.hyper_params["min_val_loss"]
        for i in range(self.hyper_params["training_iterations"]):
            self._deep_template.train() # Gpytorch training mode of deep kernel model
            self.likelihood.train() # Gpytorch training mode of likelihood

            optimizer.zero_grad() # zero backprop gradients
            outputs = self._deep_template(self.train_x,self.full_train_i) # get output from model
            train_loss = -mll(outputs, self.train_y.reshape(-1)) # calculate loss and backprop derivatives
            train_loss.backward()
            optimizer.step()
            print("training loss on training epochs", str(i)+":", train_loss)

            self._deep_template.eval() # Gpytorch evaluation mode of deep kernel model
            self.likelihood.eval() # Gpytorch evaluation mode of likelihood
            with torch.no_grad():
                preds = self._deep_template(self.val_x, self.full_val_i)
                val_loss = -mll(preds, self.val_y.reshape(-1))

            if min_loss > val_loss and val_loss > 0:
                min_loss = val_loss
                self._model_params = copy.deepcopy(self._deep_template.state_dict())
            else:
                counter += 1
                if counter % self.hyper_params["decay_gap"] == 0:
                    for params in optimizer.param_groups:
                        params['lr'] *= self.hyper_params["decay_rate"]
            if counter >= self.hyper_params["patience"]:
                break

        # retrain to select suitable kernel
        self.kernel = self.kernel_selection()
        self._deep_template.covar_module = self.kernel.cuda() # reset the selected kernel
        optimizer = torch.optim.Adam(self._deep_template.parameters(), lr=0.3*self.hyper_params["learning_rate"]) # decrease the learning rate
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self._deep_template) # reset the loss function

        counter = 0 # record the account of iterations that fail to decrease the validation loss
        min_loss = self.hyper_params["min_val_loss"]
        for i in range(self.hyper_params["training_iterations"]):
            self._deep_template.train() # Gpytorch training mode of deep kernel model
            self.likelihood.train() # Gpytorch training mode of likelihood

            optimizer.zero_grad() # zero backprop gradients
            outputs = self._deep_template(self.train_x,self.full_train_i) # get output from model
            train_loss = -mll(outputs, self.train_y.reshape(-1)) # calculate loss and backprop derivatives
            train_loss.backward()
            optimizer.step()
            print("training loss on retraining epochs", str(i)+":", train_loss)

            self._deep_template.eval() # Gpytorch evaluation mode of deep kernel model
            self.likelihood.eval() # Gpytorch evaluation mode of likelihood
            with torch.no_grad():
                preds = self._deep_template(self.val_x, self.full_val_i)
                val_loss = -mll(preds, self.val_y.reshape(-1))

            if min_loss > val_loss and val_loss > 0:
                min_loss = val_loss
                self._model_params = copy.deepcopy(self._deep_template.state_dict())
            else:
                counter += 1
                if counter % self.hyper_params["decay_gap"] == 0:
                    for params in optimizer.param_groups:
                        params['lr'] *= self.hyper_params["decay_rate"]
            if counter >= self.hyper_params["patience"]:
                break

    def predict(self):
        # Making Predictions
        self._deep_template.load_state_dict(self._model_params) # load parameters performence the best on validation set
        self._deep_template.eval() # Gpytorch evaluation mode of deep kernel model
        self.likelihood.eval() # Gpytorch evaluation mode of likelihood

        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            for i in range(self.hyper_params["pre_horizon"]):
                if i == 0:
                    test_x = self.test[i:-self.hyper_params["pre_horizon"]+i,:self.hyper_params["seq_len"]].clone()
                test_y = self.test[i:-self.hyper_params["pre_horizon"]+i,self.hyper_params["seq_len"]].squeeze().clone()

                preds = self._deep_template(test_x,self.full_test_i)
                if i == 0:
                    predictions = preds.mean.reshape(-1,self.hyper_params["num_nodes"]).unsqueeze(1)
                else:
                    predictions = torch.cat([predictions, preds.mean.reshape(-1,self.hyper_params["num_nodes"]).unsqueeze(1)], axis=1)

                test_x = torch.cat([test_x.squeeze().clone(), preds.mean.reshape(-1,self.hyper_params["num_nodes"]).unsqueeze(1)], axis=1)
                test_x = test_x[:,1:,:].clone()

            predictions = predictions.cpu()
            for i in range(self.hyper_params["num_nodes"]):
                predictions[:,:,i] = predictions[:,:,i]*self.std[i]+self.mean[i]
            return predictions

    def dump_model(self):
        torch.save(self._deep_template, "./saved_model/AutoGP_weights.pth")
        return self._model_params

    def load_model(self, model_params):
        self._deep_template.load_state_dict(model_params)
