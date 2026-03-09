import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# 1️⃣ Path to test dataset
# -------------------------
test_dir = 'dataset/test'   # ✅ keep this IF your dataset folder exists

# -------------------------
# 2️⃣ Load trained model (FIXED PATH)
# -------------------------
model = tf.keras.models.load_model('cnn_model.h5')
print("✅ Model loaded successfully")

# -------------------------
# 3️⃣ Prepare test data
# -------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# -------------------------
# 4️⃣ Evaluate model
# -------------------------
test_loss, test_accuracy = model.evaluate(test_generator)

print("✅ Test Accuracy:", test_accuracy)
print("✅ Test Loss:", test_loss)