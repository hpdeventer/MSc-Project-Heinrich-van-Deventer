import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Embedding, Reshape, RepeatVector, Multiply, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
import matplotlib.pyplot as plt
import numpy as np

# Define the target function
def f(x1, x2):
    return np.sin(4 * np.pi * x1) * np.sin(4 * np.pi * x2)

def create_partitions(n_partitions):
    # Create a list to store the partitions' limits
    partitions = []

    # Create equally sized intervals in [0, 1]^2
    for i in range(n_partitions):
        for j in range(n_partitions):
            partitions.append([(i/n_partitions, (i+1)/n_partitions), (j/n_partitions, (j+1)/n_partitions)])

    # Randomly shuffle the order of the partitions
    random.shuffle(partitions)

    return partitions

def generate_training_data(partitions, n_samples):
    # Create a list to store all training data and labels
    all_X_train = []
    all_y_train = []

    # For each partition generate training points and train all models on them
    for idx, ((x1_min, x1_max), (x2_min, x2_max)) in enumerate(partitions):
        # Generate training data within the current partition and evaluate the function at these points
        X_train_partition = np.random.uniform([x1_min, x2_min], [x1_max, x2_max], (n_samples, 2))
        y_train_partition = f(X_train_partition[:, 0], X_train_partition[:, 1])

        # Append the current partition's data to all_X_train and all_y_train 
        all_X_train.append(X_train_partition)
        all_y_train.append(y_train_partition)

    # Concatenate all training data and labels into numpy arrays
    all_X_train = np.concatenate(all_X_train, axis=0)
    all_y_train = np.concatenate(all_y_train, axis=0)

    return all_X_train, all_y_train

def plot_training_data(partitions,X,y,n_samples, plot_name='', save=False):
    
    n_partitions = 4

    values = np.linspace(0, 1, n_partitions**2)
    colormap = plt.cm.viridis
    colors = colormap(values)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for idx in range(len(partitions)):
        # Plot the sampled points for visual confirmation
        ax.scatter(X[idx*n_samples:(idx+1)*n_samples, 0], X[idx*n_samples:(idx+1)*n_samples, 1], color=colors[idx % len(colors)], alpha=0.82, s=10., label='Partition '+str(idx+1))
        ax.text((partitions[idx][0][0]+partitions[idx][0][1])/2,(partitions[idx][1][0]+partitions[idx][1][1])/2,str(idx+1), color='black',ha='center',va='center',weight='bold', fontsize=12,bbox=dict(facecolor='white', edgecolor='none'))

    ax.set_xlabel('$x_1$') 
    ax.set_ylabel('$x_2$')

    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if save:
        plt.savefig(f'{plot_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def predict_models(model_list):
    
    # Create a grid of points in the domain [0, 1]^2
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    X_test = np.stack([X1.flatten(), X2.flatten()]).T
    
    # List to store all predictions
    all_pred = []

    for (model, name) in model_list:
        print(f"Predicting with {name}...")
        y_pred = model.predict(X_test)

        # Reshape the prediction array to have the same shape as X1 and X2
        Z_pred = y_pred.reshape(X1.shape)
        
        # Store Z_pred in list
        all_pred.append(Z_pred)

    return all_pred

def plot_predictions(model_list, predictions, plot_name='', save=False):
    fig, axs = plt.subplots(1, len(model_list), figsize=(len(model_list)*3, 5), sharex=True, sharey=True)
    
    vmin = -1.0 #min([pred.min() for pred in predictions])
    vmax = +1.0 #max([pred.max() for pred in predictions])

    for (model, name), ax, Z_pred in zip(model_list, axs.flatten(), predictions):
    
        # Plot the predicted function using imshow with shared color limits
        im = ax.imshow(Z_pred, extent=[0, 1, 0 ,1], origin='lower', cmap='viridis', vmin=vmin,vmax=vmax)
        
        ax.set_title(name)

    for ax in axs:
        ax.set_xlabel('$x_1$') 

    axs[0].set_ylabel('$x_2$')  # only set y-label for the first subplot

    # Create a divider for the existing axes instance.
    divider = make_axes_locatable(ax)

    # Add an axes to the right of the main axes.
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add a colorbar to the last subplot with adjusted height
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    
    if save:
        plt.savefig(f'{plot_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def pseudorehearsal(input_dim: int, num_samples: int, 
                    model: tf.keras.Model, 
                    train_x: np.ndarray, 
                    train_y: np.ndarray, 
                    seed_val: int) -> tuple:
    '''
    Generate pseudorehearsal samples using a given model, combine them with the original training data, 
    and return the shuffled combined dataset.

    Args:
    - input_dim (int): Dimension of the input space.
    - num_samples (int): Number of pseudorehearsal samples to generate.
    - model (tf.keras.Model): Model to generate rehearsal targets.
    - train_x (np.ndarray): Original training input data.
    - train_y (np.ndarray): Original training target data.
    - seed_val (int): Seed value for random number generation.

    Returns:
    - tuple: Shuffled combined input and target data.
    '''
    
    # Generate pseudorehearsal samples, assume uniform over [0,1]^n
    rng = np.random.default_rng(seed_val)
    rehearsal_samples = rng.uniform(0, 1, size=(num_samples, input_dim))
    #print("rehearsal starts")
    rehearsal_targets = model(rehearsal_samples)
    #print("ends")
    
    # Combine training data with pseudorehearsal data and shuffle them
    combined_x = np.concatenate([train_x, rehearsal_samples], axis=0)
    combined_y = np.concatenate([train_y, rehearsal_targets], axis=0)
    
    indices = np.arange(combined_x.shape[0])
    rng.shuffle(indices)

    shuffled_x = combined_x[indices]
    shuffled_y = combined_y[indices]

    return shuffled_x, shuffled_y

def initialize_all_models(input_dimension: int, 
                          seed_val: int, 
                          output_dim: int = 1,
                          hidden_units_wide: int = 1000,
                          hidden_units_deep: int = 16,
                          hidden_layers: int = 8,
                          num_exps: int = 6) -> list:
    """Initialize models with given configurations."""
    common_args = {
        'input_dim': input_dimension, 
        'output_dim': output_dim, 
        'seed': seed_val
    }

    models = [
        (create_wide_relu_ann(hidden_units=hidden_units_wide, **common_args), "Wide ReLU ANN"),
        (create_deep_relu_ann(hidden_units=hidden_units_deep, hidden_layers=hidden_layers, **common_args), "Deep ReLU ANN"),
    ]

    for partition_num in [20]:
        models.append((SplineANN(partition_num=partition_num, **common_args), 
                       f"Spline ANN (z={partition_num})"))
        models.append((ABELSpline(partition_num=partition_num, num_exps=num_exps, **common_args), 
                       f"ABEL-Spline (z={partition_num})"))
        models.append((LookupTableModel(partition_num=partition_num, default_val=-1., **common_args),
                       f"Lookup Table (z={partition_num})"))

    return models

def compile_models(models, optimizer='adam', loss='mean_absolute_error'):
    """Compile TensorFlow/Keras models."""
    for model, name in models:
        model.compile(optimizer=optimizer, loss=loss)


def create_linear_model(input_dim: int, output_dim: int = 1, seed: int = 42) -> Sequential:
    """Create a linear model with rescaling and a dense layer.
    
    Args:
        input_dim: The input dimension.
        output_dim: The output dimension. Defaults to 1.
        seed: The seed for deterministic weight initialization. Defaults to 42.
        
    Returns:
        A Sequential model consisting of the linear layers.
    """
    initializer = keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(Rescaling(scale=2., offset=-1., input_shape=(input_dim,)))
    model.add(Dense(output_dim, kernel_initializer=initializer))
    return model

def create_wide_relu_ann(input_dim: int, hidden_units: int, output_dim: int = 1, seed: int = 42) -> Sequential:
    """Create a wide ReLU activated artificial neural network.
    
    Args:
        input_dim: The input dimension.
        hidden_units: The number of hidden units.
        output_dim: The output dimension. Defaults to 1.
        seed: The seed for deterministic weight initialization. Defaults to 42.
        
    Returns:
        A Sequential model consisting of the ANN layers.
    """
    initializer = keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(Rescaling(scale=2., offset=-1., input_shape=(input_dim,)))
    model.add(Dense(hidden_units, activation='relu', kernel_initializer=initializer))
    model.add(Dense(output_dim, kernel_initializer=initializer))
    return model

def create_deep_relu_ann(input_dim: int, hidden_units: int, hidden_layers: int, output_dim: int = 1, seed: int = 42) -> Sequential:
    """Create a deep ReLU activated artificial neural network.
    
    Args:
        input_dim: The input dimension.
        hidden_units: The number of hidden units.
        hidden_layers: The number of hidden layers.
        output_dim: The output dimension. Defaults to 1.
        seed: The seed for deterministic weight initialization. Defaults to 42.
        
    Returns:
        A Sequential model consisting of the ANN layers.
    """
    initializer = keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(Rescaling(scale=2., offset=-1., input_shape=(input_dim,)))
    for _ in range(hidden_layers):
        model.add(Dense(hidden_units, activation='relu', kernel_initializer=initializer))
    model.add(Dense(output_dim, kernel_initializer=initializer))
    return model 

class LookupTableModel(tf.keras.Model):
    def __init__(self, input_dim: int, partition_num: int, output_dim: int = 1,
                 default_val: float = 0.0, seed: int = 55):
        super(LookupTableModel, self).__init__()
        self.input_dim = input_dim
        self.partition_num = partition_num
        initializer = tf.keras.initializers.RandomUniform(seed=seed)
        self.embedding = tf.keras.layers.Embedding(partition_num**input_dim + 1, output_dim,
                                                   embeddings_initializer=initializer)
        self.default_val = tf.constant(default_val, dtype=tf.float32)

        # Set last entry in embedding to be default value
        self.embedding.build((None,))
        self.embedding.set_weights([tf.concat([self.embedding.weights[0].numpy()[:-1],
                                               [[default_val]*output_dim]], axis=0)])
        
        # Changed to integer type
        self.partition_num_powers = tf.cast(tf.pow(partition_num, tf.range(input_dim)), dtype=tf.int32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # ReLU operation to drop negative inputs 
        inputs = tf.maximum(0., inputs)

        # Scale, floor and cast to integers
        scaled_input = tf.cast(tf.floor(inputs * self.partition_num), dtype=tf.int32)

        # Bounding the indices by partition number - 1
        bounded_inputs = tf.minimum(scaled_input, self.partition_num - 1)

        # Flatten each vector to get a single index for each sample.
        indices = tf.reduce_sum(bounded_inputs * self.partition_num_powers, axis=1)

        
        outputs = self.embedding(indices)
        return outputs

class ABELSpline(keras.Model):
    """
    ABELSpline Class for Anti-Symmetric Exponential Spline Additive Neural Network.
    """
    def __init__(self, input_dim: int, partition_num: int, num_exps: int, output_dim: int, seed: int = 55, **kwargs):
        """
        Initialize the ABELSpline model.

        :param input_dim: Dimension of the input data
        :param partition_num: Number of partitions
        :param num_exps: Number of exponential terms
        :param output_dim: Output dimension
        """
        super(ABELSpline, self).__init__(**kwargs)
        
        # Setting up the model parameters
        self.input_dim, self.partition_num, self.num_exps, self.output_dim = input_dim, partition_num, num_exps, output_dim
        
        # Direct Spline Additive Neural Network (SAM)
        self.direct_sam = SplineANN(input_dim=self.input_dim, 
                                                      output_dim=self.output_dim, 
                                                      partition_num=self.partition_num,
                                                      seed=hash("Direct: " + str(seed)) % (2**32))
        
        # Anti-Symmetric Exponential layer, if there are exponential terms
        if self.num_exps > 0:
            self.anti_symmetric_exponential_layer = AntiSymmetricExponential(num_exps=num_exps, output_dim=output_dim)
            self.indirect_sam = SplineANN(input_dim=input_dim,
                                                            output_dim=int(2*num_exps*output_dim), 
                                                            partition_num=partition_num,
                                                            seed=hash("Indirect: " + str(seed)) % (2**32))

    def call(self, inputs):
        """
        Forward pass of the model.

        :param inputs: Input tensor
        :return: Output tensor
        """
        output_accumulator = self.direct_sam(inputs)
        
        # If there are exponential terms, incorporate them into the output
        if self.num_exps > 0:
            spline_additive_output = self.indirect_sam(inputs)
            output_anti_symmetric_exponential = self.anti_symmetric_exponential_layer(spline_additive_output)
            output_accumulator = tf.keras.layers.Add()([output_accumulator, output_anti_symmetric_exponential])
        
        return output_accumulator

    def repartition(self, new_partition_num):
        """
        Create a new ABELSpline model with a different number of partitions.

        :param new_partition_num: Number of partitions for the new model
        :return: A new instance of the ABELSpline model
        """
        # Creating a new model with the new partition number
        new_model = ABELSpline(input_dim=self.input_dim, partition_num=new_partition_num, num_exps=self.num_exps, output_dim=self.output_dim)
        new_model.build(input_shape=(None,self.input_dim))
        
        # Transferring weights from old to new model
        w1 = self.indirect_sam.repartition(new_partition_num).get_weights()
        new_model.indirect_sam.set_weights(w1)
        w2 = self.direct_sam.repartition(new_partition_num).get_weights()
        new_model.direct_sam.set_weights(w2)
        del w1, w2
        
        return new_model
        
class AntiSymmetricExponential(tf.keras.layers.Layer):
    def __init__(self, num_exps, output_dim, **kwargs):
        super(AntiSymmetricExponential, self).__init__(**kwargs)
        self.num_exps = num_exps
        self.output_dim = output_dim
        self.bias_val = tf.constant(-2.*tf.math.log(tf.range(0.,num_exps)+1.), dtype=tf.float32)
        self.reshape_layer = tf.keras.layers.Reshape((self.output_dim, 2 ,self.num_exps))
        self.reshape_output = tf.keras.layers.Reshape((self.output_dim,))
    
    def call(self, inputs):
        reshaped_inputs = self.reshape_layer(inputs)
        add_bias = tf.nn.bias_add(reshaped_inputs, self.bias_val)
        exponentials = tf.math.exp(add_bias)
        summed = tf.reduce_sum(exponentials,axis=-1 ,keepdims=False)
        list_of_exponentials = tf.split(summed,num_or_size_splits=2,axis=-1)
        difference = tf.keras.layers.subtract(list_of_exponentials)
        output = self.reshape_output(difference)
        
        return output
    
def cubic_spline(x: tf.Tensor) -> tf.Tensor:
    """
    Generates a cubic spline for a given Tensor.

    :param x: Input tensor
    :return: Output tensor with cubic spline transformation
    """
    conditions = [tf.math.logical_and(i <= x, x < i + 1) for i in range(4)]
    
    polynomials = [
        x**3/6,
        (-3.*(x-1.)**3 +3.*(x-1.)**2 + 3*(x-1.)+1.)/6.,
        (3*(x-2)**3 - 6*(x-2)**2 + 4. )/6.,
        ( 4. -x)**3/6.
    ]
    zeros = tf.zeros_like(x)
    return tf.reduce_sum(tf.stack([tf.where(cond, poly, zeros) for cond, poly in zip(conditions, polynomials)]), axis=0)

def floormod_activation(x: tf.Tensor) -> tf.Tensor:
    """Applies floor modulus 1 to a given Tensor."""
    return tf.math.floormod(x, 1.)

class SplineANN(keras.Model):
    def __init__(self, input_dim: int, output_dim: int, partition_num: int,  seed: int = 55, **kwargs):
        super(SplineANN, self).__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.density = 4*partition_num + 3
        self.input_dimension_shift = tf.repeat(tf.range(0., self.input_dim, dtype=tf.float32) * self.density, 4)
        self.reshape_input = Reshape((self.input_dim,1), name="Reshape_Input")
        self.scale_floormod = self._create_conv1d_layer(1, floormod_activation, self.density - 3, "Scale_and_Floormod")
        self.cubic_spline = self._create_conv1d_layer(4, cubic_spline, 1., "Cubic_Spline", bias=3 - np.arange(0, 4))
        self.reshape_splines = Reshape((self.input_dim * 4,),name="Reshape_Splines")
        self.repeat_splines = RepeatVector(self.output_dim, name="Repeat_Splines")
        self.floor_shift = self._create_conv1d_layer(4, tf.math.floor, self.density - 3, "Floor_and_Shift", bias=np.arange(0,4), dtype=tf.float32)        
        self.reshape_ints = Reshape((self.input_dim * 4,), name="Reshape_Ints")
        #self.control_points = self._create_control_points()
        self.control_points = self._create_control_points(seed)

    def call(self, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        reshaped_input = self.reshape_input(input_tensor)
        spline_values = self.reshape_splines(self.cubic_spline(self.scale_floormod(reshaped_input)))
        #transposed_splines = tf.transpose(self.repeat_splines(spline_values), perm=[0, 2, 1])
        transposed_splines = tf.transpose(self.repeat_splines(spline_values), perm=[0, 2, 1])
        floor_shift_values = self.reshape_ints(self.floor_shift(reshaped_input)) 
        floor_div_values = tf.math.floormod(floor_shift_values, self.density)
        adjusted_input_dim_values = tf.nn.bias_add(floor_div_values, self.input_dimension_shift)
        control_points_values = self.control_points(adjusted_input_dim_values)
        #print(control_points_values.shape)
        #return tf.math.reduce_sum(Multiply()([transposed_splines,control_points_values]),1, keepdims=False)
        return tf.math.reduce_sum(Multiply()([transposed_splines,control_points_values]),-2, keepdims=False)

    def _create_control_points(self, seed : int) -> Embedding:
        return Embedding(self.input_dim * self.density, 
                         self.output_dim, 
                         input_length=self.input_dim, 
                         #embeddings_initializer='uniform',
                         embeddings_initializer=keras.initializers.RandomUniform(seed=seed),
                         trainable=True, 
                         name="Control_Points")

    def _create_conv1d_layer(self, filters: int, activation: tf.Tensor, kernel: float, name: str, bias: float = None, dtype: tf.DType = None) -> Conv1D:
        kernel_initializer = tf.constant_initializer(kernel)
        bias_initializer = tf.constant_initializer(bias) if bias is not None else None
        return Conv1D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=1,
            activation=activation,
            use_bias=bias is not None,
            trainable=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            dtype=dtype,
            name=name
        )

    def construct(self) -> None:
        self(tf.keras.layers.Input(shape=(self.input_dim,)))
        self.call(keras.Input(shape=(self.input_dim,)))
        #self.build(input_shape=tuple(self.input_dim,))
        
        
    # every output dimension has its own dimension... That is why the shape is not correct
    def repartition(self, partition_num):
        old_weights = self.control_points.get_weights()[0]
        new_model = SplineANN(self.input_dim, self.output_dim, partition_num)
        #new_model.construct()
        new_model.build(input_shape=(None,self.input_dim)) #change did not fix warning
        new_weights = new_model.control_points.get_weights()[0]
        density = 4 * partition_num + 3
        knots   = -(1. - np.arange(0., (density)))/(density - 3.)
        knots = np.reshape(knots, (density,1)) # change
        model = tf.keras.Sequential([
                layers.Reshape((1, 1), input_shape=(1,), name="Reshape_Input"),
                layers.Conv1D(filters=density,  # 4 * partition_num + 3 = 4 * 50 + 3 = 203
                              kernel_size=1,
                              strides=1,
                              padding='valid',
                              data_format='channels_last',
                              dilation_rate=1,
                              activation=cubic_spline,
                              use_bias=True,
                              trainable=False,
                              kernel_initializer=tf.constant_initializer(float(density-3.)),
                              bias_initializer=tf.constant_initializer(3. - np.arange(0., (density))),
                              dtype=tf.float32,
                              name="conv1d_spline")
                ])
        model.build(input_shape=(None, 1));
        M = model.predict(knots).reshape(density,density) 
        del model
        for i in range(0,self.input_dim):
            for j in range(0,self.output_dim):
                dense_weights = old_weights[i*self.density:(i+1)*self.density,j]
                target_model = keras.Sequential([
                    layers.Reshape((1, 1), input_shape=(None,)),
                    layers.Conv1D(filters=self.density,
                                  kernel_size=1,
                                  strides=1,
                                  padding='valid',
                                  data_format='channels_last',
                                  dilation_rate=1, 
                                  activation=cubic_spline,
                                  use_bias=True,
                                  trainable=False,
                                  kernel_initializer=tf.constant_initializer(float(self.density - 3.)),
                                  bias_initializer=tf.constant_initializer(3. - np.arange(0., self.density)),
                                  dtype=tf.float32),
                    layers.Flatten(),
                    layers.Dense(1, activation='linear', kernel_initializer=tf.constant_initializer(dense_weights))
                ])
                function_values = target_model.predict(knots).reshape(len(knots))
                coefficients = np.linalg.solve(M,function_values)
                new_weights[i*density:(i+1)*density,j] = coefficients
                del target_model, function_values, coefficients
        del knots, density
        new_model.control_points.set_weights([new_weights])
        return new_model
