import numpy as np
import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

def build_mnist( ):
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data( )
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

    model = tf.keras.models.Sequential([
        tf.keras.Input((28,28,1)),
        tf.keras.layers.Conv2D(filters=12, kernel_size=(3,3),
                             activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(filters=18, kernel_size=(3,3),
                             activation=tf.nn.relu),

        tf.keras.layers.MaxPool2D(pool_size=(3,3)),

        tf.keras.layers.Flatten( ),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(10)
        ] )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                                                         from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=8)

    model.evaluate(x_test,  y_test, verbose=2)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[i], cmap='gray', vmin=0, vmax=1)
        batch = np.array( [ x_test[i] ] )
        prediction = tf.math.argmax(model.predict(batch)[0])
        plt.title(str(int(prediction)))
        plt.axis('off')
    plt.show()

    return model

if __name__ == '__main__':
    model = build_mnist()

    modelPath = 'mnist-model.keras'
    model.save(modelPath)
    