import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Visualize the dataset
plt.figure(figsize=(10, 8))
sns.countplot(train_labels)
fig, axes = plt.subplots(ncols=5, sharex=False, sharey=True, figsize=(10, 4))
for i in range(5):
    axes[i].set_title(train_labels[i])
    axes[i].imshow(train_images[i], cmap='gray_r')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()

# Pre-process the data
print('Training images shape:', train_images.shape)
print('Testing images shape:', test_images.shape)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Apply normalization
train_images = train_images / 255.0
test_images = test_images / 255.0
num_classes = 10

# Create a Sequential model
model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3),
                 activation=tf.nn.relu,
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(32, (3, 3), activation=tf.nn.relu))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation=tf.nn.softmax))

model.summary()

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=train_images,
                    y=train_labels,
                    validation_split=0.1,
                    epochs=10)

# Save the trained model
model.save('MNIST_model.h5')

from tensorflow.keras.models import load_model
model = load_model('MNIST_model.h5')

# Evaluate the model on the test data
loss_and_acc = model.evaluate(test_images, test_labels)
print("Test Loss", loss_and_acc[0])
print("Test Accuracy", loss_and_acc[1])

# Visualize training and validation metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].plot(epochs, acc, 'y', label='Training accuracy')
ax[0].plot(epochs, val_acc, 'g', label='Validation accuracy')
ax[0].legend(loc=0)
ax[1].plot(epochs, loss, 'y', label='Training loss')
ax[1].plot(epochs, val_loss, 'g', label='Validation loss')
ax[1].legend(loc=0)

plt.suptitle('Training and validation')
plt.show()

# Create and visualize the confusion matrix
y_predicted = model.predict(test_images)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
confusion_matrix = tf.math.confusion_matrix(labels=test_labels, predictions=y_predicted_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Test the Model on Specific Examples
def test_example(index):
    plt.imshow(test_images[index], cmap='gray_r')
    plt.title('Actual Value: {}'.format(test_labels[index]))
    prediction = model.predict(test_images)
    plt.axis('off')
    predicted_value = np.argmax(prediction[index])
    print('Predicted Value:', predicted_value)
    if test_labels[index] == predicted_value:
        print('Successful prediction')
    else:
        print('Unsuccessful prediction')

# Test on two examples
test_example(7)
test_example(1)
