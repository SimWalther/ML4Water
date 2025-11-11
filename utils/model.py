import tensorflow as tf
import configparser
import keras
import pydot
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint
from keras.saving import register_keras_serializable
import keras_tuner


class Model:
    def __init__(
        self,
        config_path="config.ini",
    ):
        config = configparser.ConfigParser()
        config.read(config_path)

        self.n_neurons_1 = int(config["Model"]["n_neurons_1"])
        self.n_neurons_2 = int(config["Model"]["n_neurons_2"])
        self.gaussian_noise = float(config["Model"]["gaussian_noise"])
        self.gaussian_noise2 = float(config["Model"]["gaussian_noise2"])
        self.dropout = float(config["Model"]["dropout"])
        self.lr = float(config["Model"]["lr"])
        self.momentum = float(config["Model"]["momentum"])
        self.loss = config["Model"]["loss"]
        self.l1_regularization = float(config["Model"]["l1_regularization"])
        self.l2_regularization = float(config["Model"]["l2_regularization"])

    def process_upstream_slavi(
        self, slavi_inputs, context_inputs, gaussian_noise, gaussian_noise2
    ):
        context = layers.GaussianNoise(stddev=gaussian_noise2)(context_inputs)

        tree = layers.GaussianNoise(stddev=gaussian_noise)(slavi_inputs)
        tree = GaussianAttention2D()(inputs=[tree, context])
        tree = layers.GlobalAveragePooling2D()(tree)
        # tree = layers.BatchNormalization()(tree)
        return tree

    def plot(self, model):
        keras.utils.plot_model(
            model,
            to_file="plots/model_architecture.png",
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=False,
            show_dtype=False,
        )

    def create(self, feature_groups, buffers, distances, upstream_context_cols):
        inputs, processed_inputs = [], []

        for (group_name, group_features), (_, group_type) in zip(
            feature_groups.groups.items(), feature_groups.groups_types.items()
        ):
            input_layer = keras.Input(shape=(len(group_features),), name=group_name)
            inputs.append(input_layer)

            if group_type == "softmax":
                processed = SoftmaxAttention1D(name=group_name + "_attention")(
                    input_layer
                )
                processed = layers.Reshape((processed.shape[-1], -1))(processed)
                processed = layers.GlobalAveragePooling1D()(processed)
            else:  # Group type 'other'
                processed = input_layer

            processed_inputs.append(processed)

        slavi_inputs = layers.Input((len(buffers), len(distances), 1), name="slavi")
        inputs.append(slavi_inputs)

        context_inputs = layers.Input((len(upstream_context_cols),), name="slope")
        inputs.append(context_inputs)

        processed_slavi = self.process_upstream_slavi(
            slavi_inputs, context_inputs, self.gaussian_noise, self.gaussian_noise2
        )
        processed_inputs.append(processed_slavi)

        x = keras.layers.concatenate(processed_inputs, name="concatenated_inputs")
        x = layers.BatchNormalization()(x)
        x = keras.layers.Dense(
            self.n_neurons_1,
            activation="silu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.L1L2(
                l1=self.l1_regularization, l2=self.l2_regularization
            ),
        )(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(
            self.n_neurons_2,
            activation="silu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.L1L2(
                l1=self.l1_regularization, l2=self.l2_regularization
            ),
        )(x)
        output = keras.layers.Dense(1)(x)
        model = keras.models.Model(inputs=inputs, outputs=output)

        model.compile(
            loss=self.loss,
            # optimizer=SGD(
            #     learning_rate=self.lr, momentum=self.momentum, weight_decay=1e-4
            # ),
            optimizer=Adam(learning_rate=self.lr),
            metrics=["mae", "mse"],
        )

        return model


class SoftmaxAttention1D(layers.Layer):
    def __init__(self, dtype="float32", name="softmaxAttention1d"):
        super(SoftmaxAttention1D, self).__init__(dtype=dtype, name=name)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="weight",
            shape=(input_shape[-1],),
            initializer=RandomNormal(mean=0.5, stddev=0.1),
            trainable=True,
            dtype=self.dtype,
        )

    def call(self, inputs):
        gamma = tf.nn.softmax(self.gamma)
        return keras.ops.multiply(inputs, gamma)

    def get_config(self):
        config = super().get_config()
        return config


class GaussianAttention2D(layers.Layer):
    def __init__(self, dtype="float32", name="gaussianAttention2d"):
        super(GaussianAttention2D, self).__init__(dtype=dtype, name=name)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("The layer expects two inputs: image and context")
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        self.context_features = input_shape[1][-1]
        self.gaussian_n_parameters = 4
        self.gaussian_weights = self.add_weight(
            name="gaussian_weights",
            shape=(
                self.context_features,
                self.gaussian_n_parameters,
            ),  # shift_x, shift_y, width_x, width_y
            initializer=RandomNormal(mean=0, stddev=0.1),
            trainable=True,
            dtype=self.dtype,
        )
        self.gaussian_bias = self.add_weight(
            name="gaussian_bias",
            shape=(self.gaussian_n_parameters,),  # shift_x, shift_y, width_x, width_y
            initializer=RandomNormal(mean=0, stddev=0.1),
            trainable=True,
            dtype=self.dtype,
        )

    def call(self, inputs):
        image_input = inputs[0]
        context_input = inputs[1]
        attention_matrix = self.attention_matrix(image_input, context_input)
        scaled_image = tf.multiply(image_input, attention_matrix)
        return scaled_image

    def attention_matrix(self, image_input, context_input):
        gaussian_params = tf.keras.activations.silu(
            tf.keras.ops.matmul(context_input, self.gaussian_weights)
            + self.gaussian_bias
        )
        width_x = gaussian_params[:, 0]
        width_y = gaussian_params[:, 1]
        shift_x = gaussian_params[:, 2]
        shift_y = gaussian_params[:, 3]

        # Enforce min and max values of gaussian parameters
        width_x = tf.keras.activations.relu(width_x) + 0.1
        width_y = tf.keras.activations.relu(width_y) + 0.1
        shift_x = tf.clip_by_value(shift_x, -0.3, 1.3)
        shift_y = tf.clip_by_value(shift_y, -0.3, 1.3)

        # Create a meshgrid for the Gaussian
        x = tf.cast(tf.linspace(0, 1, self.width), dtype="float32")
        y = tf.cast(tf.linspace(0, 1, self.height), dtype="float32")

        # Repeat the meshgrid for each batch
        x = tf.tile(x[tf.newaxis, :], [tf.shape(image_input)[0], 1])
        y = tf.tile(y[tf.newaxis, :], [tf.shape(image_input)[0], 1])

        # Compute Gaussian values for each dimension
        gaussian_x = self.gaussian(x, shift_x[:, tf.newaxis], width_x[:, tf.newaxis])
        gaussian_y = self.gaussian(y, shift_y[:, tf.newaxis], width_y[:, tf.newaxis])

        # Multiply the Gaussian distributions to get 2D kernels
        attention_matrix = tf.einsum("ij,ik->ijk", gaussian_y, gaussian_x)
        attention_matrix = tf.expand_dims(attention_matrix, axis=-1)

        # Normalize
        total_weights = tf.reduce_sum(attention_matrix, axis=[1, 2], keepdims=True)
        attention_matrix = attention_matrix / total_weights
        return attention_matrix

    def gaussian(self, x, shift, width):
        return tf.exp(-(((x - shift) ** 2) / (2 * width**2)))

    def get_config(self):
        config = super(GaussianAttention2D, self).get_config()
        config.update({"width": self.width, "height": self.height})
        return config
