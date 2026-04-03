from TrainingToy import train

# Main script
from pathlib import Path

# Run the simulation
script_path = Path(__file__).resolve()
script_dir = script_path.parent

# Data directory for datasets
data_dir = ...
datasets = ['RevDigital']

# Number of epochs
EPOCHS = 100000

# Initial learning rate
LR = 3e-4

# Possible model types and their respective minibatch and batch sizes
model_ts = ['fft', 'iir', 'fft_real', 'iir_real', 'ir']
SEQ_LEN = 48000*2
BATCH_SIZE = 1
# Filter length (model units)
filter_length = 48000

for dataset in datasets:
    for model_t in model_ts:
        name = 'stft' + model_t

        train(data_dir=data_dir,
                save_folder=dataset + name,
                dataset=dataset,
                seq_len=SEQ_LEN,
                cond_dim=2,
                batch_size=BATCH_SIZE,
                lr=LR,
                filter_length=filter_length,
                epochs=EPOCHS,
                model_t=model_t,
                dry_wet=False
              )
