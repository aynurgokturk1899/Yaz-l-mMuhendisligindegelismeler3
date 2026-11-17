"""
CNN Ödevi - Fashion-MNIST
Bölümler: 3 (Konvolüsyon), 4 (Pooling), 5 (ReLU/Softmax),
          6 (CNN modeli), 7 (Mimari), 8 (Deney)
Colab'de veya yerelde çalıştırılabilir. Görseller PNG olarak kaydedilir.
"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)

# ------------------------------------------------------------------
# 3. BÖLÜM: Konvolüsyon
# ------------------------------------------------------------------
print("=" * 60, "\n3. BÖLÜM: KONVOLÜSYON\n", "=" * 60)
conv = tf.keras.layers.Conv2D(
    filters=16, kernel_size=(3, 3), activation="relu",
    kernel_initializer="he_normal",
)
dummy_input = tf.random.normal((1, 128, 128, 1))
output = conv(dummy_input)
weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)
print("Çıkış boyutu:", output.shape)
print("Filtre ağırlıklarının boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)
print("Toplam parametre:", weights.size + biases.size)
print("İlk filtre:\n", weights[:, :, 0, 0])

# Madde 7: tüm filtrelerin başlangıç ağırlıklarını görselleştir
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
vmax = np.abs(weights).max()
for i, ax in enumerate(axes.flat):
    ax.imshow(weights[:, :, 0, i], cmap="gray", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Filtre {i + 1}", fontsize=9)
    ax.axis("off")
plt.suptitle("16 filtrenin başlangıç ağırlıkları (3x3, he_normal)")
plt.tight_layout()
plt.savefig("01_filtre_agirliklari.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# 4. BÖLÜM: Max Pooling
# ------------------------------------------------------------------
print("\n", "=" * 60, "\n4. BÖLÜM: MAX POOLING\n", "=" * 60)
input_data = tf.constant([[
    [1., 3., 2., 4.],
    [5., 6., 1., 2.],
    [7., 2., 8., 3.],
    [4., 1., 5., 9.],
]])
input_data = tf.reshape(input_data, (1, 4, 4, 1))
pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
pooled = pool(input_data)

print("Giriş (4x4):\n", input_data.numpy()[0, :, :, 0])
print("Pooling sonucu (2x2):\n", pooled.numpy()[0, :, :, 0])
print("Elle hesap: [[6, 4], [7, 9]]")

# ------------------------------------------------------------------
# 5. BÖLÜM: ReLU ve Softmax
# ------------------------------------------------------------------
print("\n", "=" * 60, "\n5. BÖLÜM: RELU VE SOFTMAX\n", "=" * 60)
x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
y = tf.keras.layers.ReLU()(x)
print("Giriş:", x.numpy())
print("ReLU çıkışı:", y.numpy())

logits = tf.constant([[2.0, 1.0, 0.1]])
probabilities = tf.keras.layers.Softmax()(logits)
print("Softmax sonuçları:", probabilities.numpy())
print("Olasılıkların toplamı:", tf.reduce_sum(probabilities).numpy())
print("En yüksek olasılıklı sınıf (argmax):",
      int(tf.argmax(probabilities, axis=1).numpy()[0]))

# ------------------------------------------------------------------
# 6. BÖLÜM: Veri seti ve CNN
# ------------------------------------------------------------------
print("\n", "=" * 60, "\n6. BÖLÜM: CNN MODELİ\n", "=" * 60)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)


def build_model(first_filters=8):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(first_filters, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


def train_and_eval(first_filters):
    tf.keras.utils.set_random_seed(42)  # adil karşılaştırma için aynı tohum
    m = build_model(first_filters)
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    start = time.time()
    h = m.fit(x_train, y_train, epochs=5, batch_size=64,
              validation_split=0.1, verbose=2)
    elapsed = time.time() - start
    t_loss, t_acc = m.evaluate(x_test, y_test, verbose=0)
    return m, h, elapsed, t_loss, t_acc


model, history, _, test_loss, test_accuracy = train_and_eval(8)
model.summary()
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# D. Eğitim grafikleri
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout()
plt.savefig("02_egitim_grafikleri.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# 7. BÖLÜM: Katman boyutları
# ------------------------------------------------------------------
print("\n", "=" * 60, "\n7. BÖLÜM: KATMAN ÇIKIŞ BOYUTLARI\n", "=" * 60)
for layer in model.layers:
    print(f"{layer.name:20s} çıkış: {layer.output.shape}  "
          f"parametre: {layer.count_params()}")

# ------------------------------------------------------------------
# 8. BÖLÜM: Deney (8 vs 16 filtre)
# ------------------------------------------------------------------
print("\n", "=" * 60, "\n8. BÖLÜM: DENEY\n", "=" * 60)
results = {}
for f in (8, 16):
    print(f"\n--- Deney: ilk katmanda {f} filtre ---")
    _, h, secs, _, t_acc = train_and_eval(f)
    results[f] = {
        "train_acc": h.history["accuracy"][-1],
        "test_acc": t_acc,
        "time": secs,
    }

print("\n{:<18}{:>12}{:>12}".format("Ölçüt", "Deney 1", "Deney 2"))
print("{:<18}{:>12}{:>12}".format("Filtre sayısı", 8, 16))
print("{:<18}{:>12.4f}{:>12.4f}".format(
    "Eğitim doğruluğu", results[8]["train_acc"], results[16]["train_acc"]))
print("{:<18}{:>12.4f}{:>12.4f}".format(
    "Test doğruluğu", results[8]["test_acc"], results[16]["test_acc"]))
print("{:<18}{:>11.1f}s{:>11.1f}s".format(
    "Eğitim süresi", results[8]["time"], results[16]["time"]))