# %%
import sys
sys.path.append('../')


# %%
# path configurations
from conf import BASE_DIR
from Applications.Poisoning.gen_configs import main as gen_configs

base_folder = BASE_DIR/'models'/'poisoning'
train_conf = BASE_DIR/'Applications'/'Poisoning'/'configs'/'demo'/'train.json'
poison_conf = BASE_DIR/'Applications'/'Poisoning'/'configs'/'demo'/'poison.json'
unlearn_conf = BASE_DIR/'Applications'/'Poisoning'/'configs'/'demo'/'unlearn.json'

# generate experiment configurations
gen_configs(base_folder, train_conf, poison_conf, unlearn_conf)


# %%
from Applications.Poisoning.train import main as train

# train one clean and one poisoned model
train(model_folder=base_folder/'clean')
train(model_folder=base_folder/'budget-5000')


# %%

# Unlearning input paths

poisoned_weights = None     # model that has been trained on poisoned data
fo_repaired_weights = None  # model weights after unlearning (first-order)
so_repaired_weights = None  # model weights after unlearning (second-order)
injector_path = None  # cached injector for reproducibility
clean_results = None  # path to reference results on clean dataset

# TODO clean_acc
clean_acc = None

# %% 

from Applications.Poisoning.model import get_VGG_CIFAR10
from Applications.Poisoning.dataset import Cifar10
from Applications.Poisoning.poison.injector import LabelflipInjector

# # load data
data = Cifar10.load()
(x_train, y_train), _, (x_val, y_val) = data
y_train_orig = y_train.copy()

# load injector
injector = LabelflipInjector.from_pickle(injector_path)

# # inject backdoors into training data
x_train, y_train = injector.inject(x_train, y_train)

# %%
