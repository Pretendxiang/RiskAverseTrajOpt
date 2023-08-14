import jax

jax_array = jax.numpy.array([1, 2, 3])
numpy_array = jax.device_get(jax_array)
print(jax_array, numpy_array)