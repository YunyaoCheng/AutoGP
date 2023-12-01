import gpytorch
from autogp import AutoGP
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

name = 1 # task_id
hyper_params = { # hyperparameters
    "num_nodes":9, #n umber of variables
    "pre_horizon":12, # length of prediction horizon
    "seq_len":12, # length of historic horizon
    "file_path":"./data/NEW-DATA.txt", # file path of data
    "cols":[
                '3:Temperature_Comedor_Sensor',
                '4:Temperature_Habitacion_Sensor',
                '5:Weather_Temperature',
                '6:CO2_Comedor_Sensor',
                '7:CO2_Habitacion_Sensor',
                '8:Humedad_Comedor_Sensor',
                '9:Humedad_Habitacion_Sensor',
                '10:Lighting_Comedor_Sensor',
                '11:Lighting_Habitacion_Sensor'
           ], #select variables
    "train_split":3200, # index for splitting the training set and test set
    "test_split":3737, # index for splitting the test set and validation set
    "moving_step":12, # the step of moving time window
    "base_kernel_set":[ # define the basic kernel set
                        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),\
                        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()),\
                        gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()),\
                        gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) # decouple to string
                      ],
    "R":1, # the number of inter-blocks
    "patch_size":[3], # the size of patches
    "input_dim":1, # the dimention of the input observation
    "output_dim":1, # the dimention of the output observation
    "device":'cuda:0', # running on this device
    "training_iterations":5, # iterations of training
    "learning_rate":0.001, # learning rate
    "min_val_loss":9999, # minimum of validation Loss
    "patience":100, # early stop steps
    "decay_gap":5, # learning ragte decay steps
    "decay_rate":0.9, # learning rate decay rate
    "numbers":1, # top-k best kernels in intra-block
}

##########################################################################################


model = AutoGP(name, hyper_params)
model.fit()
predictions = model.predict()
weights = model.dump_model()
model.load_model(weights)

print(predictions)
