from tensorflow.keras.layers import Layer, Dense
from keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable(package="Custom")
class SASelfAttentionLayer(Layer):
    def __init__(self, units, **kwargs):
        super(SASelfAttentionLayer, self).__init__(**kwargs)
        self.units = units
        # These will be defined in build()
        self.W = None
        self.V = None

    def build(self, input_shape):
        # Define the Dense layers with correct input dimensions
        self.W = Dense(self.units)
        self.V = Dense(1)
        super(SASelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Self-attention computation
        score = self.V(tf.nn.tanh(self.W(inputs)))          # (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)    # (batch_size, time_steps, 1)
        context_vector = attention_weights * inputs         # (batch_size, time_steps, features)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, features)
        return context_vector  # or return context_vector, attention_weights

    def get_config(self):
        config = super(SASelfAttentionLayer, self).get_config()
        config.update({
            "units": self.units,
        })
        return config
