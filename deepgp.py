import gpytorch
from patch_attention import PA

class AutoDKL(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, hyper_params, kernel):
        super(AutoDKL, self).__init__(train_x, train_y, likelihood)
        self.hyper_params = hyper_params
        self.mean_module = gpytorch.means.ConstantMean() # mean function
        self.covar_module = kernel # kernel function
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=self.hyper_params["num_nodes"], rank=1) # cross-variable weights
        self.feature_extractor = PA(horizon=1, lag=self.hyper_params["seq_len"], dynamic=False, supports=None, # patch_attention
                                    patch_sizes=self.hyper_params["patch_size"], channels=32,
                                    num_nodes=self.hyper_params["num_nodes"], input_dim=self.hyper_params["input_dim"],
                                    output_dim=self.hyper_params["output_dim"], device=self.hyper_params["device"])

    def forward(self, x, index):
        projected_x, _ = self.feature_extractor(x)
        projected_x = projected_x.squeeze().reshape(-1)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        covar_i = self.task_covar_module(index)
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
