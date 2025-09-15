from TrainingRev import train

"""
main script

"""

# number of epochs
EPOCHS = 100000
# batch size
# initial learning rate
LR = 3e-4
# number of model's units
filter_length = 48000
datasets = ['Reverb48']

# data_dir: the directory in which datasets are stored
# data_dir = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/SSM_VA/Pickles/'########
data_dir = '../../Files/Kernels/'
cond_type = None
MINI_BATCH_SIZEs = [48000*2, 48000*2, 48000, 48000*2, 48000, 480]
model_ts = ['ir', 'fft', 'lstm', 'tcn', 'ssm', 'iir']
BATCH_SIZEs = [1000, 1000, 1, 1, 1, 1]

MINI_BATCH_SIZEs = [48000*2, 48000*2, 48000*2]
model_ts = ['ir', 'fft', 'iir']
BATCH_SIZEs = [1, 1, 1]

units = 32
for dataset in datasets:
    for model_t, MINI_BATCH_SIZE, BATCH_SIZE in zip(model_ts, MINI_BATCH_SIZEs, BATCH_SIZEs):
        name = 'time' + model_t

        train(data_dir=data_dir,
                save_folder=dataset + name,
                dataset=dataset,
                mini_batch_size=MINI_BATCH_SIZE,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                filter_length=filter_length,
                epochs=EPOCHS,
                units=units,
                model_t=model_t,
                cond_type=cond_type,
                dry_wet=True,
                freq_loss=False,
                time_loss=True,
                inference=False)
