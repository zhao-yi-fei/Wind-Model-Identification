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
X = np.load('E:\\updraft\\updraft_wind_samples1000.npy')  # shape=(1000,100,100,100,3)
print(X.shape)
y_params = np.load('E:\\updraft\\updraft_wind_param1000.npy')  # shape=(1000,4)

# split data
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

# save normalization parameters
np.save('E:\\updraft\\train_min_per_channel.npy', min_per_channel)
np.save('E:\\updraft\\train_max_per_channel.npy', max_per_channel)

# Check if each channel has all-zero or constant values
for channel in range(3):
    print(f"Channel {channel}: min={X_train[..., channel].min()}, max={X_train[..., channel].max()}")

# --------------------- Parameter normalization ---------------------
params_min = np.zeros((1, 4))
params_max = np.zeros((1, 4))

# V_0, x_0, y_0, z_0
for p in range(4):
    params_min[0,0] = y_params_train[:,0].min()
    params_max[0,0] = y_params_train[:,0].max()   

# save normalization parameters
np.save('E:\\updraft\\train_min_params.npy', params_min)
np.save('E:\\updraft\\train_max_params.npy', params_max)

y_params_train_normalized = np.zeros_like(y_params_train)
y_params_val_normalized = np.zeros_like(y_params_val)

# training set normalization
for i in range(len(y_params_train)):
    for p in range(4):
        min_v = params_min[0, p]
        max_v = params_max[0, p]
        if max_v - min_v < 1e-8:
            y_params_train_normalized[i, p] = 0.0
        else:
            y_params_train_normalized[i, p] = (y_params_train[i, p] - min_v) / (max_v - min_v)
    
# validation set normalization
for i in range(len(y_params_val)):
    for p in range(4):
        min_v = params_min[0, p]
        max_v = params_max[0, p]
        if max_v - min_v < 1e-8:
            y_params_val_normalized[i, p] = 0.0
        else:
            y_params_val_normalized[i, p] = (y_params_val[i, p] - min_v) / (max_v - min_v)
    
print("Normalized parameter dimensions:", y_params_train_normalized.shape)  # Should output (700, 4)
print("Training set parameter range:", y_params_train_normalized.min(), y_params_train_normalized.max())
print("Validation set parameter range:", y_params_val_normalized.min(), y_params_val_normalized.max())

def create_model(input_shape=(100, 100, 100, 3)):
    """
    Create multi-head model for predicting V_core, x_0, y_0, and z_0.
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
    
    # v head: V_core
    v = Dense(32, activation='swish')(shared)
    v_output = Dense(1, activation='linear', name='v_output')(v)
    
    # xy head: x_0, y_0
    xy = Dense(64, activation='swish')(shared)
    xy_output = Dense(2, activation='linear', name='xy_output')(xy)

    # z0 head: z_0
    z0 = Dense(32, activation='swish')(shared)
    z0_output = Dense(1, activation='linear', name='z0_output')(z0)

    outputs=[v_output, xy_output, z0_output]

    return Model(inputs, outputs)

def plot_combined_training_history(history, save_dir, show=True):
    """
    Plot total loss (training/validation) and training/validation loss curves for each output and save the image.
    Layout: 2x2 subplots, font uses Times New Roman, parameter names displayed in LaTeX format.
    """

    hist = history.history
    any_key = next(iter(hist))
    epochs = np.arange(1, len(hist[any_key]) + 1)

    # Calculate total loss for training and validation
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

    # Font properties
    fp = FontProperties(family='Times New Roman', size=11)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_total = axes[0, 0]
    ax_v = axes[0, 1]
    ax_xy = axes[1, 0]
    ax_z0 = axes[1, 1]

    # Marker size
    msize = 3
    
    # total loss
    ax_total.plot(epochs, train_total, label='Training Loss', marker='o', markersize=msize, linewidth=1)
    if val_total is not None and val_total.size == train_total.size and val_total.sum() != 0:
        ax_total.plot(epochs, val_total, label='Validation Loss', marker='o', linewidth=1, markersize=msize)
    ax_total.set_title('(a) Total Loss', fontproperties=fp)
    ax_total.set_xlabel('Epoch', fontproperties=fp)
    ax_total.set_ylabel('Loss', fontproperties=fp)
    ax_total.legend(prop=fp)
    ax_total.grid(alpha=0.3)

    # v output (V_core)
    train_key_v = 'v_output_loss'
    val_key_v = 'val_v_output_loss'
    if train_key_v in hist:
        ax_v.plot(epochs, hist[train_key_v], label=f'Training Loss', marker='o', linewidth=1, markersize=msize)
    if val_key_v in hist:
        ax_v.plot(epochs, hist[val_key_v], label=f'Validation Loss', marker='o', linewidth=1, markersize=msize)
    ax_v.set_title(r'(b) $V_{core}$ Loss', fontproperties=fp)
    ax_v.set_xlabel('Epoch', fontproperties=fp)
    ax_v.set_ylabel('Loss', fontproperties=fp)
    ax_v.legend(prop=fp)
    ax_v.grid(alpha=0.3)

    # xy output (x_0, y_0)
    train_key_xy = 'xy_output_loss'
    val_key_xy = 'val_xy_output_loss'
    if train_key_xy in hist:
        ax_xy.plot(epochs, hist[train_key_xy], label=f'Training Loss', marker='o', linewidth=1, markersize=msize)
    if val_key_xy in hist:
        ax_xy.plot(epochs, hist[val_key_xy], label=f'Validation Loss', marker='o', linewidth=1, markersize=msize)
    ax_xy.set_title(r'(c) $x_0,\ y_0$ Loss', fontproperties=fp)
    ax_xy.set_xlabel('Epoch', fontproperties=fp)
    ax_xy.set_ylabel('Loss', fontproperties=fp)
    ax_xy.legend(prop=fp)
    ax_xy.grid(alpha=0.3)

    # z0 output (z_0)
    train_key_z0 = 'z0_output_loss'
    val_key_z0 = 'val_z0_output_loss'
    if train_key_z0 in hist:
        ax_z0.plot(epochs, hist[train_key_z0], label=f'Training Loss', marker='o', linewidth=1, markersize=msize)
    if val_key_z0 in hist:
        ax_z0.plot(epochs, hist[val_key_z0], label=f'Validation Loss', marker='o', linewidth=1, markersize=msize)
    ax_z0.set_title(r'(d) $z_0$ Loss', fontproperties=fp)
    ax_z0.set_xlabel('Epoch', fontproperties=fp)
    ax_z0.set_ylabel('Loss', fontproperties=fp)
    ax_z0.legend(prop=fp)
    ax_z0.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # Save directory
    save_dir = "E:\\updraft"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   
    y_v_train = y_params_train_normalized[:, 0]         # V_core 
    y_v_test  = y_params_val_normalized[:,  0]
    
    y_xy_train = y_params_train_normalized[:, 1:3]      # x_0, y_0
    y_xy_test  = y_params_val_normalized[:, 1:3]

    y_z0_train = y_params_train_normalized[:, 3]        # z_0
    y_z0_test  = y_params_val_normalized[:,  3]

    # create model
    model = create_model(input_shape=(100, 100, 100, 3))
    
    # compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            'v_output': 'mse',
            'xy_output': 'mse',
            'z0_output': 'mse',      
        },
        loss_weights={
            'v_output': 2.5,
            'xy_output': 1.0,          
            'z0_output': 1.0,
        },
        metrics={
            'v_output': ['mae'],
            'xy_output': ['mae'],
            'z0_output': ['mae'],
        }
    )

    # ------- training -------
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
        os.path.join(save_dir, 'updraft_best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    history = model.fit(
        X_train,
        {
            'v_output': y_v_train,
            'xy_output':  y_xy_train,
            'z0_output': y_z0_train,
        },
        validation_data=(
            X_val,
            {
                'v_output': y_v_test,
                'xy_output':  y_xy_test,
                'z0_output': y_z0_test,
            }
        ),
        epochs=200,
        batch_size=8,
        callbacks=[early_stopping, lr_scheduler, model_checkpoint],
        verbose=1
    )

    model.save(os.path.join(save_dir, 'updraft_model.h5'))
    
    plot_combined_training_history(history, save_dir, show=True)
    print("\nModel saved!")
