import numpy as np
import json
import os
import tensorflow as tf

# 載入模型並提取權重和架構
model = tf.keras.models.load_model('fashion_mnist.h5')
params = {}
arch = []

for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        params[f"{layer.name}/kernel:0"] = weights[0]
        params[f"{layer.name}/bias:0"] = weights[1]
    
    arch.append({
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": layer.get_config(),
        "weights": [f"{layer.name}/kernel:0", f"{layer.name}/bias:0"] if weights else []
    })

# 儲存檔案
os.makedirs('model', exist_ok=True)
np.savez('model/fashion_mnist.npz', **params)
with open('model/fashion_mnist.json', 'w') as f:
    json.dump(arch, f, indent=2)
