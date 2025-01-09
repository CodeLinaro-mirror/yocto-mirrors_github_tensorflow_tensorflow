# Gemma2 2B Keras model

Scripts to run Gemma2 2B Keras model on CPU.

Model link:
https://www.kaggle.com/models/google/gemma-2/keras

Instructions:

* Set up your Kaggle API key by following [these instructions](https://www.kaggle.com/docs/api#authentication).
* `bash setup.sh`
* `KERAS_BACKEND=jax bash run.sh`
    * Set `KERAS_BACKEND=tensorflow` or `torch` to run with TensorFlow or
      Pytorch backend.
