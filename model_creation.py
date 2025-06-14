from keras import layers, models, optimizers

# A simple baseline CNN model with two Conv2D layers and a dense layer. Good for small datasets or quick experimentation.
def create_model_v1(img_size):
    model = models.Sequential([
        layers.Input(shape=(*img_size, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# A deeper CNN model with BatchNormalization and GlobalAveragePooling. More robust than v1, better suited for more complex features.
def create_model_v2(img_size):
    model = models.Sequential([
        layers.Input(shape=(*img_size, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model



# Advanced CNN model with skip (residual-like) connections and projection layers. Inspired by ResNet-style connections for better feature reuse and gradient flow.
def create_model_v3(img_size):
    inputs = layers.Input(shape=(*img_size, 3))

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)

    # Block 2
    x1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D(2, 2)(x1)

    # Block 3 (with skip connection)
    x2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D(2, 2)(x2)

    # Projection for skip connection
    x1_proj = layers.Conv2D(128, (1, 1), strides=2, padding='same')(x1)  # Match size and channels
    x = layers.Add()([x2, x1_proj])  # Residual-like connection

    # Block 4
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model