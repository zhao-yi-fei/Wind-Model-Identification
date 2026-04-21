from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, BatchNormalization, Input
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
from keras.utils import Sequence
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

# load data
X = np.load('E:\\classification\\wind_samples_class.npy') # shape=(300, 100, 100, 100, 3)
y_class = np.load('E:\\classification\\wind_onehot_class.npy')   # shape=(300,3)

# split data into training and validation sets
X_train, X_val, y_class_train, y_class_val = train_test_split(
    X, y_class, test_size=0.3, random_state=42)

# --------------------- Wind speed data normalization ---------------------
# Calculate min and max for each channel in training set
min_per_channel = X_train.min(axis=(0, 1, 2, 3))  
max_per_channel = X_train.max(axis=(0, 1, 2, 3))  

# Avoid division by zero for channels with constant values
epsilon = 1e-8
max_per_channel = np.where(max_per_channel == min_per_channel, max_per_channel + epsilon, max_per_channel)

# Normalize training and validation sets
X_train = 2 * (X_train - min_per_channel) / (max_per_channel - min_per_channel) - 1
X_val = 2 * (X_val - min_per_channel) / (max_per_channel - min_per_channel) - 1

# Print the range of normalized data to verify it is in [-1, 1]
print("Training set range:", X_train.min(), X_train.max())
print("Validation set range:", X_val.min(), X_val.max())

# Save normalization parameters for future test set use
np.save('E:\\classification\\train_min_per_channel.npy', min_per_channel)
np.save('E:\\classification\\train_max_per_channel.npy', max_per_channel)

# Check if each channel has all-zero or constant values
for channel in range(3):
    print(f"通道 {channel}: min={X_train[..., channel].min()}, max={X_train[..., channel].max()}")

def create_model(input_shape=(100,100,100,3)):
    """
    Create a 3D convolutional neural network model for multi-class classification.
    Args:
        input_shape: Shape of the input data
    Returns:
        Model: A compiled Keras model ready for training.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', name='Conv3D_1')(inputs)
    x = BatchNormalization(name='BatchNorm_1')(x) 
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='MaxPool3D_1')(x)
    
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same', name='Conv3D_2')(x)
    x = BatchNormalization(name='BatchNorm_2')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='MaxPool3D_2')(x)
    
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same', name='Conv3D_3')(x)
    x = BatchNormalization(name='BatchNorm_3')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='MaxPool3D_3')(x)
    
    x = Flatten(name='Flatten')(x)
    x = Dense(256, activation='relu', name='Dense_1')(x)
    
    class_output = Dense(3, activation='softmax', name='class_output')(x)
    
    # create model
    model = Model(inputs=inputs, outputs=class_output)
    
    # compile model
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),  
        loss={
            'class_output': 'categorical_crossentropy'
        },
        loss_weights={
            'class_output': 1.0                                                          
        },
        metrics={
            'class_output': 'accuracy'
        }
    )
    
    return model

class DataGenerator(Sequence):
    def __init__(self, X, y_class, batch_size):
        self.X = X
        self.y_class = y_class
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.X))
        X_batch = self.X[start:end]
        y_class_batch = self.y_class[start:end]
        return X_batch, {'class_output': y_class_batch}

def plot_training_history(history, save_dir):
    """
    Plot training and validation loss and accuracy curves for classification model.
    """
    # Font settings (Times New Roman)
    fp = FontProperties(family='Times New Roman', size=12)

    hist = history.history

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot loss curve
    if 'loss' in hist:
        ax1.plot(hist['loss'], label='Training Loss', marker='o', markersize=3, linewidth=1)
    if 'val_loss' in hist:
        ax1.plot(hist['val_loss'], label='Validation Loss', marker='o', markersize=3, linewidth=1)
    ax1.set_title('Classification Loss', fontproperties=fp)
    ax1.set_xlabel('Epochs', fontproperties=fp)
    ax1.set_ylabel('Loss', fontproperties=fp)
    ax1.legend(prop=fp)
    ax1.grid(alpha=0.3)

    # Plot accuracy curve
    if 'accuracy' in hist:
        ax2.plot(hist['accuracy'], label='Training Accuracy', marker='o', markersize=3, linewidth=1)
    if 'val_accuracy' in hist:
        ax2.plot(hist['val_accuracy'], label='Validation Accuracy', marker='o', markersize=3, linewidth=1)
    ax2.set_title('Classification Accuracy', fontproperties=fp)
    ax2.set_xlabel('Epochs', fontproperties=fp)
    ax2.set_ylabel('Accuracy', fontproperties=fp)
    ax2.legend(prop=fp)
    ax2.grid(alpha=0.3)

    # Set tick label fonts to Times New Roman
    for ax in (ax1, ax2):
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fp)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    # save directory for model and plots
    save_dir = "E:\\classification"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model = create_model()
    
    model.summary()
    
    # define data generators for training and validation
    batch_size = 32
    train_generator = DataGenerator(X_train, y_class_train, batch_size)
    val_generator = DataGenerator(X_val, y_class_val, batch_size)
    
    X_batch, y_batch = train_generator[0]

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
        os.path.join(save_dir, 'classification_best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=[early_stopping,lr_scheduler,model_checkpoint],  
        verbose=1
    )
    
    plot_training_history(history, save_dir)
    
    model.save(os.path.join(save_dir, "classification_model.h5"))
    print("\nModel saved!")