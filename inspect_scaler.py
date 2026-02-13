import numpy as np

data = np.load('./models/model_20260208_200152/production_model/feature_scaler.npz', allow_pickle=True)
print('All keys:', list(data.keys()))
for k in data.keys():
    print(f'{k}: {data[k]}')
    print(f'  type: {type(data[k])}')
    if hasattr(data[k], 'shape'):
        print(f'  shape: {data[k].shape}')
