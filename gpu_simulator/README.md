### Basic Environment Setup

In order to create a minimal installation with mjx and its dependencies run
the following command:

conda create --name <env> --file requirements.txt

The environment is built for CUDA-12.6 which is the latest version of drivers from NVIDIA. If CUDA version does not match, consider either upgrading CUDA version or installing brax and jax manually using pip. Test the driver version using `nvidia-smi`.

### Testing GPU Backend

Run the following code after setting up the environment:

```python
import jax
jax.default_backend()
```

If the output is 'gpu' the setup has been successful.




