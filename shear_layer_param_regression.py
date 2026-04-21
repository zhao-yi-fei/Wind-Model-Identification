import tensorflow as tf
from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D, Dense,Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import GlobalAveragePooling3D

# data loading
X = np.load('E:\\shear-layer\\shear_layer_wind_samples1000.npy')  # shape=(1000,100,100,200,3)
X = X[:,::2,::2,::2,:]  #  shape=(1000,50,50,100,3) 
print(X.shape)
y_params = np.load('E:\\shear-layer\\shear_layer_wind_param1000.npy')   # shape=(1000,5)

# Split into training and validation sets
X_train, X_val, y_params_train, y_params_val = train_test_split(
    X, y_params, test_size=0.3, random_state=42)

# --------------------- Wind speed data normalization ---------------------
# Calculate min and max for each channel in training set
min_per_channel = X_train.min(axis=(0, 1, 2, 3))  
max_per_channel = X_train.max(axis=(0, 1, 2, 3))  

# Avoid division by zero error (if a channel has identical data)
epsilon = 1e-8
max_per_channel = np.where(max_per_channel == min_per_channel, max_per_channel + epsilon, max_per_channel)

# Normalize training and validation sets
X_train = 2 * (X_train - min_per_channel) / (max_per_channel - min_per_channel) - 1
X_val = 2 * (X_val - min_per_channel) / (max_per_channel - min_per_channel) - 1

# Check if normalized range is within [-1, 1]
print("Training set range:", X_train.min(), X_train.max())
print("Validation set range:", X_val.min(), X_val.max())

# Save normalization parameters for future test set use
np.save('E:\\shear-layer\\train_min_per_channel.npy', min_per_channel)
np.save('E:\\shear-layer\\train_max_per_channel.npy', max_per_channel)

# Check if each channel has all-zero or constant values
for channel in range(3):
    print(f"Channel {channel}: min={X_train[..., channel].min()}, max={X_train[..., channel].max()}")

# --------------------- Parameter normalization ---------------------
params_min = np.zeros((1, 5))
params_max = np.zeros((1, 5))

# V_0, V_r, Δh, h_0 are normalized based on their min and max in the training set
for p in range(5):
    params_min[0, p] = y_params_train[:, p].min()
    params_max[0, p] = y_params_train[:, p].max()
# For ψ, we know the range is [0, 2π], so we can set it directly to avoid outliers affecting normalization
params_min[0,4] = 0
params_max[0,4] = 2*np.pi

# Save normalization parameters
np.save('E:\\shear-layer\\train_min_params.npy', params_min)
np.save('E:\\shear-layer\\train_max_params.npy', params_max)

# Normalize each parameter for each category separately
y_params_train_normalized = np.zeros_like(y_params_train)
y_params_val_normalized = np.zeros_like(y_params_val)

# Training set normalization
for i in range(len(y_params_train)):
    for p in range(5):
        min_v = params_min[0, p]
        max_v = params_max[0, p]
        if max_v - min_v < 1e-8:
            y_params_train_normalized[i, p] = 0.0
        else:
            y_params_train_normalized[i, p] = (y_params_train[i, p] - min_v) / (max_v - min_v)

# Validation set normalization
for i in range(len(y_params_val)):
    for p in range(5):
        min_v = params_min[0, p]
        max_v = params_max[0, p]
        if max_v - min_v < 1e-8:
            y_params_val_normalized[i, p] = 0.0
        else:
            y_params_val_normalized[i, p] = (y_params_val[i, p] - min_v) / (max_v - min_v) 

print("Normalized parameter dimensions:", y_params_train_normalized.shape)  # Should output (700, 5)
print("Training set parameter range:", y_params_train_normalized.min(), y_params_train_normalized.max())
print("Validation set parameter range:", y_params_val_normalized.min(), y_params_val_normalized.max())

def degree_mae(y_true, y_pred):
    """
    Calculate mean absolute error for ψ
    Args:
        y_true: True values
        y_pred: Predicted values
    Returns:
        mae: Mean absolute error
    """
    diff = (tf.abs(y_true - y_pred) * 360) % 360  # Convert to degrees and handle cyclic nature
    cyclic_diff = tf.minimum(diff, 360 - diff)  
    return cyclic_diff

def sc_mse(y_true, y_pred):
    """
    Calculate mean squared error for ψ
    Args:
        y_true: True values
        y_pred: Predicted values
    Returns:
        mse: Mean squared error
    """
    sc_mse = tf.reduce_mean((tf.square(tf.sin(y_true*2*np.pi) - tf.sin(y_pred*2*np.pi)) + 
                             tf.square(tf.cos(y_true*2*np.pi) - tf.cos(y_pred*2*np.pi)))/2, axis=-1)
    return sc_mse

def create_model(input_shape=(50, 50, 100, 3)):
    """
    Create multi-head model for predicting V_0, V_r, Δh, h_0, and ψ
    Args:
        input_shape: Shape of input data
    Returns:
        model: Keras Model object    
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    x = Conv3D(16, (3, 3, 3), padding='same', activation='swish')(inputs)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(32, (3, 3, 3), padding='same', activation='swish')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='swish')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = GlobalAveragePooling3D()(x)
    shared = Dense(128, activation='swish', name='shared_dense')(x)

    # v head: predict V_0 and V_r
    v = Dense(64, activation='swish')(shared)
    v_output = Dense(2, activation='linear', name='v_output')(v)

    # h head: predict Δh and h_0
    h = Dense(64, activation='swish')(shared)
    h_output = Dense(2, activation='linear', name='h_output')(h)

    # psi head: predict ψ
    psi = Dense(32, activation='swish')(shared)
    psi_output = Dense(1, activation='linear', name='psi_output')(psi)

    outputs=[v_output, h_output, psi_output]
  
    return Model(inputs, outputs)
    
def plot_combined_training_history(history, save_dir, show=True):
    """
    Plot total loss (training/validation) and training/validation loss curves for each output and save the image.
    Layout: 2x2 subplots, font uses Times New Roman, parameter names displayed in LaTeX format.
    """  

    hist = history.history
    any_key = next(iter(hist))
    epochs = np.arange(1, len(hist[any_key]) + 1)

    # Calculate total loss (training/validation)
    if 'loss' in hist:
        train_total = np.array(hist['loss'])
    else:
        train_total = np.zeros(len(epochs))
        for k, v in hist.items():
            if k.endswith('_loss') and not k.startswith('val_'):
                train_total += np.array(v)

    if 'val_loss' in hist:
        val_total = np.array(hist['val_loss'])
    else:
        val_total = np.zeros(len(epochs))
        for k, v in hist.items():
            if k.startswith('val_') and k.endswith('_loss'):
                val_total += np.array(v)

    # Font settings (Times New Roman)
    fp = FontProperties(family='Times New Roman', size=11)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_total = axes[0, 0]
    ax_v = axes[0, 1]
    ax_h = axes[1, 0]
    ax_psi = axes[1, 1]

    # Plot point size
    msize = 3
    
    # Subplot: Total loss
    ax_total.plot(epochs, train_total, label='Training Loss', marker='o', markersize=msize, linewidth=1)
    if val_total is not None and val_total.size == train_total.size and val_total.sum() != 0:
        ax_total.plot(epochs, val_total, label='Validation Loss', marker='o', linewidth=1, markersize=msize)
    ax_total.set_title('(a) Total Loss', fontproperties=fp)
    ax_total.set_xlabel('Epoch', fontproperties=fp)
    ax_total.set_ylabel('Loss', fontproperties=fp)
    ax_total.legend(prop=fp)
    ax_total.grid(alpha=0.3)

    # v output (V_0, V_r)
    train_key_v = 'v_output_loss'
    val_key_v = 'val_v_output_loss'
    if train_key_v in hist:
        ax_v.plot(epochs, hist[train_key_v], label=f'Training Loss', marker='o', linewidth=1, markersize=msize)
    if val_key_v in hist:
        ax_v.plot(epochs, hist[val_key_v], label=f'Validation Loss', marker='o', linewidth=1, markersize=msize)
    ax_v.set_title(r'(b) $V_0,\ V_r$ Loss', fontproperties=fp)
    ax_v.set_xlabel('Epoch', fontproperties=fp)
    ax_v.set_ylabel('Loss', fontproperties=fp)
    ax_v.legend(prop=fp)
    ax_v.grid(alpha=0.3)

    # h output (Delta_h, h_0)
    train_key_h = 'h_output_loss'
    val_key_h = 'val_h_output_loss'
    if train_key_h in hist:
        ax_h.plot(epochs, hist[train_key_h], label=f'Training Loss', marker='o', linewidth=1, markersize=msize)
    if val_key_h in hist:
        ax_h.plot(epochs, hist[val_key_h], label=f'Validation Loss', marker='o', linewidth=1, markersize=msize)
    ax_h.set_title(r'(c) $\Delta h,\ h_0$ Loss', fontproperties=fp)
    ax_h.set_xlabel('Epoch', fontproperties=fp)
    ax_h.set_ylabel('Loss', fontproperties=fp)
    ax_h.legend(prop=fp)
    ax_h.grid(alpha=0.3)

    # psi output
    train_key_psi = 'psi_output_loss'
    val_key_psi = 'val_psi_output_loss'
    if train_key_psi in hist:
        ax_psi.plot(epochs, hist[train_key_psi], label=f'Training Loss', marker='o', linewidth=1, markersize=msize)
    if val_key_psi in hist:
        ax_psi.plot(epochs, hist[val_key_psi], label=f'Validation Loss', marker='o', linewidth=1, markersize=msize)
    ax_psi.set_title(r'(d) $\psi$ Loss', fontproperties=fp)
    ax_psi.set_xlabel('Epoch', fontproperties=fp)
    ax_psi.set_ylabel('Loss', fontproperties=fp)
    ax_psi.legend(prop=fp)
    ax_psi.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # Create save directory
    save_dir = "E:\\shear-layer"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    y_v_train = y_params_train_normalized[:, 0:2]      # V_0, V_r
    y_v_val  = y_params_val_normalized[:, 0:2]
    
    y_h_train = y_params_train_normalized[:, 2:4]      # delta_h, h_0
    y_h_val  = y_params_val_normalized[:,  2:4]
    
    y_psi_train = y_params_train_normalized[:, 4:5]    # ψ
    y_psi_val  = y_params_val_normalized[:,  4:5]

    # Create model
    model = create_model(input_shape=(50, 50, 100, 3))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            'v_output': 'mse',
            'h_output': 'mse',
            'psi_output': sc_mse,      
        },
        loss_weights={
            'v_output': 1.0,
            'h_output': 5.0,   # Slightly increase h weight to encourage better learning of h parameters
            'psi_output': 1.0,
        },
        metrics={
            'v_output': ['mae'],
            'h_output': ['mae'],
            'psi_output': [degree_mae], 
        }
    )

    model.summary()

    # ------- Training -------

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        os.path.join(save_dir, 'shear_layer_best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    history = model.fit(
        X_train,
        {
            'v_output': y_v_train,
            'h_output':  y_h_train,
            'psi_output': y_psi_train,
        },
        validation_data=(
            X_val,
            {
                'v_output': y_v_val,
                'h_output':  y_h_val,
                'psi_output': y_psi_val,
            }
        ),
        epochs=200,
        batch_size=8,
        callbacks=[early_stopping, lr_scheduler, model_checkpoint],
        verbose=1
    )

    model.save(os.path.join(save_dir, 'shear_layer_model.h5'))

    plot_combined_training_history(history, save_dir, show=True)
    print("\nModel saved!")