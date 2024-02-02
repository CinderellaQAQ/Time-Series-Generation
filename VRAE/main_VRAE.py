## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from VRAE import train
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
# from metrics.discriminative_metrics import discriminative_score_metrics
# from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


## Data loading
data_name = 'stock'
seq_len = 48

if data_name in ['stock', 'energy']:
    ori_data = real_data_loading(data_name, seq_len)
elif data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, seq_len, dim)
print(data_name + ' dataset is ready.')

## Newtork parameters
parameters = dict()
parameters['iterations'] = 1000
parameters['device'] = 'cuda:0'
parameters['batch_size'] = 128
parameters['seq_len'] = seq_len
parameters['feature_dim'] = ori_data[0].shape[1]
parameters['hidden_dim'] = 48
parameters['num_layer'] = 3
parameters['learning_rate'] = 1e-3
parameters['gamma'] = 0.1

# Run TimeGAN
generated_data = train(ori_data, parameters)
print('Finish Synthetic Data Generation')

from matplotlib import pyplot as plt
fig, axs = plt.subplots(4, generated_data[0].shape[1], figsize=(16, 10))
for i in range(1000, 1004):
    for j in range(generated_data[0].shape[1]):
        axs[i % 4, j].plot(ori_data[i][:, j], 'red')
        axs[i % 4, j].plot(generated_data[i][:, j], 'blue')
        axs[i % 4, j].legend(['ground truth', 'generate'])

## Performance metrics
# 1. Visualization (PCA and tSNE)
visualization(ori_data, generated_data, 'pca')
visualization(ori_data, generated_data, 'tsne')

