from keras import models
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/piotr/PycharmProjects/cifarML/output",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(32, 32),
    batch_size=128)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      "/home/piotr/PycharmProjects/cifarML/cifar/test",
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(32, 32),
      batch_size=128)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

layer = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(32, 32, 3),
    pooling='max',
    classes=10,
)

model = models.Sequential()
model.add(layer)
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.build([None, 32, 32, 3])

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=30, batch_size=128,
                    validation_data=val_ds)

# accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
