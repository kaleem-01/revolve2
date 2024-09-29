# newer version of mujoco
pip install mujoco==3.2.3

# install mjx
pip install mujoco-mjx

# add jaxlib for cuda (cuda 11 for older hardware)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install brax
pip install brax
