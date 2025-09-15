from Training import train

"""
main script

"""

# number of epochs
EPOCHS = 100000
# initial learning rate
LR = 3e-4
# number of model's units
filter_length = 1024

data_dir = '../../Files/Kernels/'

model_ts = ['ir', 'fft', 'lstm', 'tcn', 'ssm', 'iir']
MINI_BATCH_SIZEs = [48000*4, 48000*4, 2400, 48000*4, 2400, 48000*4]
BATCH_SIZEs = [1, 1, 1, 1, 1, 1, 1]

datasets = ['Filtersall']

for dataset in datasets:
        for model_t, MINI_BATCH_SIZE, BATCH_SIZE in zip(model_ts, MINI_BATCH_SIZEs, BATCH_SIZEs):

            name = 'time'+ cond_type + model_t

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
                    freq_loss=False,
                    time_loss=True,
                    inference=False)
