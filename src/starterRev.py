from Training import train

"""
main script

"""

# number of epochs
EPOCHS = 1#100000
# batch size
MINI_BATCH_SIZE = 48000*2
# initial learning rate
LR = 3e-4
# number of model's units
filter_length = 48000

# data_dir: the directory in which datasets are stored
# data_dir = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/SSM_VA/Pickles/'########
data_dir = '../../Files/Kernels/'
cond_type = None
datasets = ['Reverb48']

for dataset in datasets:
      name = 'freq'
      train(data_dir=data_dir,
            save_folder=dataset + name,
            dataset=dataset,
            mini_batch_size=MINI_BATCH_SIZE,
            learning_rate=LR,
            filter_length=filter_length,
            epochs=EPOCHS,
            cond_type=cond_type,
            dry_wet=True,
            freq_loss=True,
            time_loss=False,
            inference=False)

      name = 'both'
      train(data_dir=data_dir,
            save_folder=dataset + name,
            dataset=dataset,
            mini_batch_size=MINI_BATCH_SIZE,
            learning_rate=LR,
            filter_length=filter_length,
            epochs=EPOCHS,
            cond_type=cond_type,
            dry_wet=True,
            freq_loss=True,
            time_loss=True,
            inference=False)

      name = 'time'

      train(data_dir=data_dir,
            save_folder=dataset + name,
            dataset=dataset,
            mini_batch_size=MINI_BATCH_SIZE,
            learning_rate=LR,
            filter_length=filter_length,
            epochs=EPOCHS,
            cond_type=cond_type,
            dry_wet=True,
            freq_loss=False,
            time_loss=True,
            inference=False)