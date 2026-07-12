
import argparse, os, sys, json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(__file__))
from dataset import load_datasets, IMG_SIZE
from model import build_model, unfreeze_top

def compute_class_weights(train_ds, num_classes):
    print("Computing class weights (this takes a minute)...")
    all_labels = []
    for _, y in train_ds.unbatch():
        all_labels.append(np.argmax(y.numpy()))
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=np.array(all_labels)
    )
    return dict(enumerate(weights))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",      default="data")
    parser.add_argument("--out",           default="models")
    parser.add_argument("--epochs-phase1", type=int,   default=10)
    parser.add_argument("--epochs-phase2", type=int,   default=20)
    parser.add_argument("--lr-phase1",     type=float, default=1e-3)
    parser.add_argument("--lr-phase2",     type=float, default=1e-5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train_ds, val_ds, class_names = load_datasets(args.data_dir)
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump({i: n for i, n in enumerate(class_names)}, f, indent=2)

    class_weights = compute_class_weights(train_ds, num_classes)
    model, base_model = build_model(num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True,
            verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out, "best_model.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    if args.epochs_phase1 > 0:
        print(f"\n{'='*55}")
        print("  Phase 1: training head only (backbone frozen)")
        print(f"{'='*55}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr_phase1),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.epochs_phase1,
            class_weight=class_weights,
            callbacks=callbacks,
        )

    if args.epochs_phase2 > 0:
        print(f"\n{'='*55}")
        print("  Phase 2: fine-tuning unfrozen backbone layers")
        print(f"{'='*55}")
        unfreeze_top(base_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr_phase2),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.epochs_phase2,
            class_weight=class_weights,
            callbacks=callbacks,
        )

    print(f"\nDone. Best model saved to {args.out}/best_model.h5")

if __name__ == "__main__":
    main()
