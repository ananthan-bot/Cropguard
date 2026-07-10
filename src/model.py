
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

def build_model(num_classes, img_size=224, dropout=0.3):
    """
    MobileNetV2 transfer learning model.

    Architecture:
        MobileNetV2 (pretrained ImageNet, frozen in Phase 1)
        → GlobalAveragePooling2D
        → Dropout(0.3)
        → Dense(num_classes, softmax)

    Why MobileNetV2?
    - Only ~2.5M parameters — small enough for a web server
    - Clean path to TFLite quantization for future mobile export
    - Strong ImageNet features transfer well to leaf texture/colour
    """
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,       # remove ImageNet classification head
        weights="imagenet",      # start from pretrained weights
    )
    base_model.trainable = False  # frozen during Phase 1

    inputs  = base_model.input
    x       = layers.GlobalAveragePooling2D()(base_model.output)
    x       = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="cropguard_mobilenetv2"), base_model

def unfreeze_top(base_model, fraction=0.7):
    """
    Unfreeze the last (1 - fraction) of backbone layers for Phase 2 fine-tuning.
    Default keeps 70% frozen and unfreezes the top 30%.

    Why not unfreeze everything?
    - Early layers detect edges/textures — useful for all vision tasks, no need to retrain
    - Later layers detect higher-level patterns — these need to adapt to leaf disease features
    - Unfreezing too much risks destroying pretrained weights with a large learning rate
    """
    base_model.trainable = True
    freeze_until = int(len(base_model.layers) * fraction)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfrozen {trainable} of {len(base_model.layers)} backbone layers")
