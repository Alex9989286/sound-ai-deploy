import tensorflow as tf
import tf2onnx
import os

print("🚀 Loading TensorFlow model...")
model = tf.keras.models.load_model("sound_class_model_mfcc_opt.keras")
print(f"✅ Model loaded")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# 定义输入签名 (batch, time, features, channels)
# 你的模型输入是 (None, 128, 248, 1)
input_signature = (tf.TensorSpec((None, 128, 248, 1), tf.float32, name="input"),)

print("🔄 Converting to ONNX...")
onnx_model, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=input_signature,
    opset=13
)

print("💾 Saving ONNX model...")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ ONNX model saved!")

# 比较文件大小
tf_size = os.path.getsize("sound_class_model_mfcc_opt.keras") / (1024 * 1024)
onnx_size = os.path.getsize("model.onnx") / (1024 * 1024)

print(f"\n📊 File size comparison:")
print(f"   TensorFlow model: {tf_size:.2f} MB")
print(f"   ONNX model: {onnx_size:.2f} MB")
print(f"   Reduced by: {(1 - onnx_size/tf_size) * 100:.1f}%")