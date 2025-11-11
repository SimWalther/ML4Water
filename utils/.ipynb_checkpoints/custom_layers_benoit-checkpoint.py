import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint





class Gaussian_Weighted_Sum(keras.layers.Layer):
    def __init__(self, units, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
 
    def build(self, input_shape):
        self.n = input_shape[-1]
        self.l = tf.keras.ops.arange(1, self.n + 1, dtype='float32')
        self.l = tf.keras.ops.reshape(self.l, (self.n, 1))
        self.l = tf.keras.ops.repeat(self.l, self.units, axis=1)
        # Gaussian mean
        self.w = self.add_weight(shape=(self.units,), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1), trainable=True)
        # Gaussian std. dev.
        self.s = self.add_weight(shape=(self.units,), initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.5), trainable=True)
        # Gaussian relative importance
        self.r = self.add_weight(shape=(self.units-1,), initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.1), trainable=True)
        # Final weight mask
        self.final_weights = self.add_weight(shape=(self.n,1), trainable=False)

    
    
        
    def call(self, inputs, training=None):
        if training:
            rel = tf.keras.ops.sigmoid(self.r)                                     # parameter normalization (sigmoid)
            rel = tf.keras.ops.concatenate([keras.ops.ones(1), rel])               # relative importance of first Gaussian is always 1
            loc = self.n * tf.keras.ops.sigmoid(self.w) + 0.5                      # parameter normalization (sigmoid * range)
            weights = tf.keras.ops.exp(-((self.l - loc) ** 2) / (self.s ** 2))     # weight computation (Gaussian)
            weights = tf.keras.ops.multiply(weights, rel)                          # weighting weights
            weights = tf.keras.ops.sum(weights, axis=-1, keepdims=True)            # weight merging (multimodal)
            weights = weights / tf.keras.ops.sum(weights, axis=0, keepdims=True)   # weight normalization (sum = 1)
            self.final_weights.assign(weights)                             # track for visualization
        else:
            weights=self.final_weights
        activation = tf.keras.ops.dot(inputs, weights)
        
        return activation
        
    def get_config(self):
        base_config = super().get_config()
        config = {'units':self.units}
        return {**base_config, **config}



# # Entropy-regularized softmax layer
# class SoftmaxWeightsDiverse(keras.layers.Layer):
#     def __init__(self, entropy_coeff=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.entropy_coeff = entropy_coeff

#     def build(self, input_shape):
#         # One trainable weight per feature in the group
#         self.kernel = self.add_weight(
#             shape=(input_shape[-1], 1),
#             initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
#             trainable=True,
#             name="kernel"
#         )

#     def call(self, inputs):
#         # Softmax to get weights between 0 and 1, summing to 1
#         w_normalized = tf.nn.softmax(self.kernel, axis=0)  # shape: (n_features, 1)

#         # Entropy regularization (maximize entropy → subtract from loss)
#         entropy = -tf.reduce_sum(w_normalized * tf.math.log(w_normalized + 1e-8))
#         self.add_loss(-self.entropy_coeff * entropy)

#         # Weighted sum of inputs
#         output = tf.linalg.matmul(inputs, w_normalized)  # shape: (batch_size, 1)
#         return output

#     def get_weights_normalized(self):
#         """Return the normalized weights for inspection"""
#         return tf.nn.softmax(self.kernel, axis=0).numpy()

#     def get_config(self):
#         base_config = super().get_config()
#         return {**base_config, "entropy_coeff": self.entropy_coeff}

# # Updated create_model
# def create_model(X_group_dict, group_definitions, n_neurons, l1_coeff, l2_coeff, dropout1, dropout2, lr, entropy_coeff=0.1,loss_to_use='mean_absolute_error'):
#     inputs, processed_inputs = [], []

#     for group_name, X_group in X_group_dict.items():
#         input_layer = keras.Input(shape=(X_group.shape[1],), name=group_name)
#         inputs.append(input_layer)

#         group_type = group_definitions.get(group_name, {}).get("type", "other")

#         if group_type == "gaussian":
#             processed = Gaussian_Weighted_Sum(units=5, name=f"gaussian_{group_name}")(input_layer)
#         elif group_type == "softmax":
#             processed = SoftmaxWeightsDiverse(entropy_coeff=entropy_coeff, name=f"softmax_{group_name}")(input_layer)
#         else:
#             processed = input_layer  # For 'input_other'

#         processed_inputs.append(processed)

#     x = keras.layers.concatenate(processed_inputs, name="concatenated_inputs")
#     x = keras.layers.Dropout(dropout1)(x)
#     x = keras.layers.Dense(
#         n_neurons,
#         activation='silu',
#         kernel_regularizer=keras.regularizers.L1L2(l1=l1_coeff, l2=l2_coeff)
#     )(x)
#     x = keras.layers.Dropout(dropout2)(x)
#     output = keras.layers.Dense(1, activation=None)(x)

#     model = keras.models.Model(inputs=inputs, outputs=output)
#     model.compile(
#         loss=loss_to_use,
#         optimizer=Adam(learning_rate=lr),
#         metrics=["mae"]
#     )

#     return model

### ARCHIVES ###

class SoftmaxWeights(keras.constraints.Constraint):
    def __init__(self, axis=0, temperature=0.5):
        self.axis = axis
        self.temperature = temperature
    
    # def __call__(self, w):
    #     w = tf.convert_to_tensor(w)
    #     e = tf.exp(w - tf.reduce_max(w))
    #     w = e / tf.reduce_sum(e)
    #     return w

    def __call__(self, w):
        # Convert w to a TensorFlow tensor (just to be safe)
        w = tf.convert_to_tensor(w)
        
        # Apply temperature scaling
        w_scaled = w / self.temperature
        
        # Subtract the max for numerical stability
        w_shifted = w_scaled - tf.reduce_max(w_scaled, axis=self.axis, keepdims=True)
        
        # Exponentiate the shifted weights
        e = tf.exp(w_shifted)
        
        # Normalize to make it sum to 1 along the specified axis
        w_normalized = e / tf.reduce_sum(e, axis=self.axis, keepdims=True)
        
        return w_normalized
    
    def get_config(self):
        return {'axis': self.axis}




# class SoftmaxWeightsDiverse(keras.constraints.Constraint):
#     def __init__(self, axis=0, temperature=0.2, diversity_strength=0.5):
#         self.axis = axis
#         self.temperature = temperature
#         self.diversity_strength = diversity_strength
    
#     def __call__(self, w):
#         # Convert to tensor
#         w = tf.convert_to_tensor(w)
        
#         # Temperature-scaled softmax
#         w_scaled = w / self.temperature
#         w_shifted = w_scaled - tf.reduce_max(w_scaled, axis=self.axis, keepdims=True)
#         e = tf.exp(w_shifted)
#         w_normalized = e / tf.reduce_sum(e, axis=self.axis, keepdims=True)
        
#         # --- Diversity encouragement ---
#         # Encourage variance among weights (not perfectly uniform)
#         # Compute deviation from uniform distribution
#         n = tf.cast(tf.shape(w_normalized)[0], tf.float32)
#         uniform = tf.ones_like(w_normalized) / n
#         diversity_penalty = self.diversity_strength * (w_normalized - uniform)
        
#         # Add diversity signal while keeping sum = 1
#         w_diverse = w_normalized + diversity_penalty
#         # Renormalize
#         w_diverse /= tf.reduce_sum(w_diverse, axis=self.axis, keepdims=True)
        
#         return w_diverse


def create_model(X_group_dict, group_definitions, n_neurons, l1_coeff, l2_coeff, dropout1, dropout2, lr,loss_to_use='mean_absolute_error'):
    inputs, processed_inputs = [], []

    for group_name, X_group in X_group_dict.items():
        input_layer = keras.Input(shape=(X_group.shape[1],), name=group_name)
        inputs.append(input_layer)

        group_type = group_definitions.get(group_name, {}).get("type", "other")

        if group_type == "gaussian":
            processed = Gaussian_Weighted_Sum(units=5, name=f"gaussian_{group_name}")(input_layer)
        elif group_type == "softmax":
            processed = keras.layers.Dense(
                1,
                use_bias=False,
                # kernel_initializer=keras.initializers.Constant(1 / X_group.shape[1]),
                kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=1),
                kernel_constraint=SoftmaxWeights(),
                # activity_regularizer=keras.regularizers.L1(0.0001),
                name=f"softmax_{group_name}"
            )(input_layer)
        else:
            processed = input_layer  # For 'input_other'

        processed_inputs.append(processed)

    x = keras.layers.concatenate(processed_inputs,name="concatenated_inputs")
    x = keras.layers.Dropout(dropout1)(x)
    x = keras.layers.Dense(n_neurons, activation='silu',
                           kernel_regularizer=keras.regularizers.L1L2(l1=l1_coeff, l2=l2_coeff))(x)
    x = keras.layers.Dropout(dropout2)(x)
    output = keras.layers.Dense(1, activation=None)(x)

    model = keras.models.Model(inputs=inputs, outputs=output)
    model.compile(loss=loss_to_use,
                  optimizer=Adam(learning_rate=lr),
                  metrics=["mae"])

    return model

