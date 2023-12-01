# Jeuralangelo
This is the [Jittor](https://github.com/Jittor/Jittor) implementation of **Neuralangelo: High-Fidelity Neural Surface Reconstruction**.
The implementation achieves nearly identical results as the official implementation, and even slightly faster and better.

# Installation
see requirements.txt

# Data preparation
Use the same dataset format with DTU dataset.

# Run Jeuralangelo
Just run main.py with your args. Configs could be modified in config.py.
After training, mesh will be automatically extracted to the log directory.

# GPU memory and run time.
Change the `fast_train` variable in `config.py` as you need.
| fast_train | GPU VRAM | Run time(RTX 4090) |
| :--------: | :------: | :-------: |
| True       | 2GB      | ~ 1 hour  |
| False      | 12GB     | ~ 4 hour  |

# Acknowledgements
The original implementation comes from the following cool project:
- [Neuralangelo](https://github.com/NVlabs/neuralangelo)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [JNeRF](https://github.com/Jittor/JNeRF)
Their licenses can be seen at licenses/, many thanks for their nice work!
