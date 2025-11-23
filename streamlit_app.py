# app.py
import streamlit as st
import os
import io
import zipfile
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import base64
import time
import shutil

# Try imports that may be optional / heavy
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, preprocessing
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False

st.set_page_config(page_title="Teachable Machine - All-in-One (single file)", layout="wide")

# ---------- Config ----------
DATA_DIR = "data"
MODELS_DIR = "models"
HISTORY_FILE = "training_history.json"
IMG_SIZE = (64, 64)
SEED = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- Utilities ----------
def save_image_to_class(img: Image.Image, class_name: str):
    class_path = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)
    # Find next index
    idx = 1
    existing = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if existing:
        nums = []
        for f in existing:
            name = os.path.splitext(f)[0]
            try:
                nums.append(int(name))
            except:
                pass
        if nums:
            idx = max(nums) + 1
    img_rgb = img.convert("RGB")
    img_rgb.save(os.path.join(class_path, f"{idx}.jpg"))

def list_classes():
    items = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]
    return items

def load_dataset(resize=IMG_SIZE):
    images, labels = [], []
    class_names = list_classes()
    for label, cname in enumerate(class_names):
        class_path = os.path.join(DATA_DIR, cname)
        for f in sorted(os.listdir(class_path)):
            if not f.lower().endswith((".jpg",".jpeg",".png")):
                continue
            try:
                img = Image.open(os.path.join(class_path, f)).convert("RGB")
                img = img.resize(resize)
                images.append(np.array(img))
                labels.append(label)
            except:
                # ignore corrupted
                pass
    if len(images)==0:
        return np.array([]), np.array([]), class_names
    return np.array(images), np.array(labels), class_names

def detect_corrupted_and_remove(show_list=False):
    removed = []
    for cname in list_classes():
        cp = os.path.join(DATA_DIR, cname)
        for f in os.listdir(cp):
            path = os.path.join(cp, f)
            try:
                Image.open(path).verify()
            except:
                removed.append(path)
                os.remove(path)
    if show_list:
        return removed
    return removed

def detect_duplicates_and_report():
    # naive duplicate detection by comparing flattened arrays
    X, y, classes = load_dataset()
    if len(X)==0:
        return []
    hashes = {}
    dup_list = []
    for i, arr in enumerate(X):
        h = arr.tobytes()
        if h in hashes:
            dup_list.append((hashes[h], i))
        else:
            hashes[h] = i
    return dup_list  # pairs (first_index, duplicate_index)

def make_zip_of_dataset():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for cname in list_classes():
            for f in os.listdir(os.path.join(DATA_DIR, cname)):
                z.write(os.path.join(DATA_DIR, cname, f), arcname=os.path.join(cname, f))
    buf.seek(0)
    return buf

def plot_confusion_matrix(cm, class_names, ax=None, normalize=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.figure
    if normalize:
        cm_sum = cm.sum(axis=1)[:, None]
        cm_norm = np.divide(cm, cm_sum, where=cm_sum!=0)
        disp = cm_norm
    else:
        disp = cm
    im = ax.imshow(disp, interpolation='nearest')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            val = disp[i,j]
            if normalize:
                txt = f"{val*100:.1f}%"
            else:
                txt = f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if val>disp.max()/2 else "black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    return fig

def save_history_entry(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

# ---------- UI ----------
st.title("🧠 Teachable Machine — All-in-One (single file)")

st.sidebar.header("Dataset / Class Management")
with st.sidebar.expander("Create new class"):
    new_class = st.text_input("New class name")
    if st.button("Create class"):
        if new_class.strip()=="":
            st.warning("Enter a class name.")
        else:
            os.makedirs(os.path.join(DATA_DIR, new_class.strip()), exist_ok=True)
            st.success(f"Class '{new_class}' created!")

classes = list_classes()
if not classes:
    st.info("No classes found. Create classes in the sidebar or upload images to start.")
else:
    st.sidebar.subheader("Modify classes")
    selected_class = st.sidebar.selectbox("Select class to manage", options=classes)
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Rename class"):
        new_name = st.sidebar.text_input("Rename to", key="rename_text")
        # use session state fallback
        if new_name:
            old = os.path.join(DATA_DIR, selected_class)
            newp = os.path.join(DATA_DIR, new_name)
            if os.path.exists(newp):
                st.sidebar.error("Target name already exists.")
            else:
                os.rename(old, newp)
                st.sidebar.success(f"Renamed '{selected_class}' to '{new_name}'")
    if col2.button("Delete class"):
        confirm = st.sidebar.checkbox(f"Confirm delete '{selected_class}'", key="confirm_delete")
        if confirm:
            shutil.rmtree(os.path.join(DATA_DIR, selected_class))
            st.sidebar.success(f"Deleted class '{selected_class}'")
            # update classes
            classes = list_classes()

# ---------------- Add images ----------------
st.header("📸 Add Images (Upload or Webcam)")

cola, colb = st.columns([2,1])
with cola:
    upload_for = st.selectbox("Select target class for uploads", options=classes if classes else ["__no_class__"])
    files = st.file_uploader("Upload images (multiple possible)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files and upload_for != "__no_class__":
        count = 0
        for f in files:
            try:
                img = Image.open(f)
                save_image_to_class(img, upload_for)
                count += 1
            except Exception as e:
                st.error(f"Failed to save {getattr(f,'name',str(f))}: {e}")
        if count>0:
            st.success(f"Saved {count} images to '{upload_for}'")
with colb:
    st.write("Or capture from webcam:")
    try:
        cam_img = st.camera_input("Take a photo")
    except Exception:
        cam_img = None
    if cam_img and upload_for != "__no_class__":
        img = Image.open(cam_img)
        save_image_to_class(img, upload_for)
        st.success(f"Saved camera image to '{upload_for}'")

# Dataset quick overview
st.header("📊 Dataset Browser & Cleaner")
X_all, y_all, class_names = load_dataset()
if len(X_all)==0:
    st.info("Dataset is empty. Upload images to proceed.")
else:
    st.write(f"Total images: **{len(X_all)}**  — Classes: **{len(class_names)}**")
    counts = {c: len(os.listdir(os.path.join(DATA_DIR,c))) for c in class_names}
    st.dataframe(pd.DataFrame.from_dict(counts, orient="index", columns=["count"]).reset_index().rename(columns={"index":"class"}))

    if st.button("Detect corrupted images and remove them"):
        removed = detect_corrupted_and_remove(show_list=True)
        st.write(f"Removed {len(removed)} corrupted images.")
        if removed:
            st.write(removed)

    if st.button("Detect duplicate images (naive)"):
        dups = detect_duplicates_and_report()
        st.write(f"Found {len(dups)} duplicates (pairs of indices in dataset). Example (first 10):")
        st.write(dups[:10])

    if st.button("Download dataset as ZIP"):
        buf = make_zip_of_dataset()
        st.download_button("Download dataset.zip", data=buf, file_name="dataset.zip", mime="application/zip")

# ---------------- Training ----------------
st.header("🧪 Train Models")

train_button = st.button("Train Now")
if train_button:
    with st.spinner("Loading and preparing dataset..."):
        X, y, class_names = load_dataset()
    if len(X)==0 or len(np.unique(y))<2:
        st.error("Need at least 2 classes with images to train.")
    else:
        # preprocess
        X = X.astype("float32") / 255.0
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        except:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        # augmentation toggle
        agg = st.checkbox("Use image augmentation during CNN training (recommended)", value=True)
        aug_gen = None
        if agg and TF_AVAILABLE:
            aug_gen = ImageDataGenerator(rotation_range=15,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         shear_range=0.05,
                                         zoom_range=0.1,
                                         horizontal_flip=True,
                                         fill_mode='nearest')
        # Flattened features for classical ML
        n_train = len(X_train)
        X_train_flat = X_train.reshape(n_train, -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        # Logistic Regression
        st.info("Training Logistic Regression (may take time)...")
        prog = st.progress(0)
        try:
            log_reg = LogisticRegression(max_iter=400, solver="saga")
            log_reg.fit(X_train_flat, y_train)
            pred_lr = log_reg.predict(X_test_flat)
            acc_lr = accuracy_score(y_test, pred_lr)
        except Exception as e:
            st.error(f"Logistic Regression failed: {e}")
            log_reg = None
            acc_lr = 0.0
            pred_lr = np.zeros(len(y_test))
        prog.progress(33)

        # Random Forest
        st.info("Training Random Forest...")
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
            rf.fit(X_train_flat, y_train)
            pred_rf = rf.predict(X_test_flat)
            acc_rf = accuracy_score(y_test, pred_rf)
        except Exception as e:
            st.error(f"Random Forest failed: {e}")
            rf = None
            acc_rf = 0.0
            pred_rf = np.zeros(len(y_test))
        prog.progress(66)

        # CNN
        if TF_AVAILABLE:
            st.info("Training CNN (TensorFlow)..")
            # small CNN
            def make_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(class_names)):
                model = models.Sequential([
                    layers.Input(shape=input_shape),
                    layers.Conv2D(32, (3,3), activation="relu"),
                    layers.MaxPooling2D(),
                    layers.Conv2D(64, (3,3), activation="relu"),
                    layers.MaxPooling2D(),
                    layers.Conv2D(128, (3,3), activation="relu"),
                    layers.MaxPooling2D(),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dropout(0.3),
                    layers.Dense(num_classes, activation="softmax")
                ])
                model.compile(optimizer="adam",
                              loss="sparse_categorical_crossentropy",
                              metrics=["accuracy"])
                return model

            cnn = make_cnn()
            epochs = st.slider("CNN epochs", min_value=3, max_value=30, value=8)
            batch = st.selectbox("Batch size", options=[8,16,32], index=1)

            # prepare dataset for TF
            # We ensure shapes are consistent
            X_train_tf = tf.convert_to_tensor(X_train)
            X_test_tf = tf.convert_to_tensor(X_test)

            if agg and aug_gen is not None:
                # using generator flow
                # create a generator from X_train,y_train
                train_gen = aug_gen.flow(X_train, y_train, batch_size=batch, shuffle=True, seed=SEED)
                steps = max(1, len(X_train) // batch)
                hist = cnn.fit(train_gen, epochs=epochs, steps_per_epoch=steps, validation_data=(X_test_tf, y_test), verbose=0)
            else:
                hist = cnn.fit(X_train_tf, y_train, epochs=epochs, batch_size=batch, validation_data=(X_test_tf, y_test), verbose=0)

            loss, acc_cnn = cnn.evaluate(X_test_tf, y_test, verbose=0)
            st.success(f"CNN test accuracy: {acc_cnn*100:.2f}%")
        else:
            st.warning("TensorFlow not available — skipping CNN training.")
            cnn = None
            acc_cnn = 0.0
            hist = None

        prog.progress(100)

        st.success("Training complete!")

        # Show metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Logistic Reg Accuracy", f"{acc_lr*100:.2f}%")
        col2.metric("Random Forest Accuracy", f"{acc_rf*100:.2f}%")
        col3.metric("CNN Accuracy", f"{acc_cnn*100:.2f}%")

        # Confusion matrix for RF (example)
        try:
            cm_rf = confusion_matrix(y_test, pred_rf)
            fig_cm = plot_confusion_matrix(cm_rf, class_names, normalize=False)
            st.subheader("Confusion Matrix (Random Forest)")
            st.pyplot(fig_cm)
        except Exception as e:
            st.warning(f"Could not plot confusion matrix: {e}")

        # classification report for RF
        try:
            st.text("Random Forest classification report:")
            st.text(classification_report(y_test, pred_rf, target_names=class_names, zero_division=0))
        except:
            pass

        # CNN training plots
        if TF_AVAILABLE and hist is not None:
            st.subheader("CNN Training Curves")
            hist_dict = hist.history
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].plot(hist_dict.get("loss", []), label="train_loss")
            ax[0].plot(hist_dict.get("val_loss", []), label="val_loss")
            ax[0].set_title("Loss")
            ax[0].legend()
            ax[1].plot(hist_dict.get("accuracy", []), label="train_acc")
            ax[1].plot(hist_dict.get("val_accuracy", []), label="val_acc")
            ax[1].set_title("Accuracy")
            ax[1].legend()
            st.pyplot(fig)

        # Save models & provide downloads
        saved_files = []
        if log_reg:
            joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic.pkl"))
            saved_files.append(os.path.join(MODELS_DIR, "logistic.pkl"))
        if rf:
            joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
            saved_files.append(os.path.join(MODELS_DIR, "random_forest.pkl"))
        if TF_AVAILABLE and cnn:
            cnn.save(os.path.join(MODELS_DIR, "cnn.h5"))
            saved_files.append(os.path.join(MODELS_DIR, "cnn.h5"))

        st.info("Models saved in 'models/' folder")
        # model download buttons
        for fpath in saved_files:
            with open(fpath, "rb") as f:
                data = f.read()
            fname = os.path.basename(fpath)
            st.download_button(f"Download {fname}", data=data, file_name=fname)

        # TFLite conversion
        if TF_AVAILABLE and cnn:
            if st.button("Convert CNN -> TFLite (and offer download)"):
                try:
                    converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
                    tflite_model = converter.convert()
                    tflite_path = os.path.join(MODELS_DIR, "cnn.tflite")
                    with open(tflite_path, "wb") as f:
                        f.write(tflite_model)
                    st.success("Converted to TFLite")
                    with open(tflite_path, "rb") as f:
                        st.download_button("Download cnn.tflite", data=f.read(), file_name="cnn.tflite")
                except Exception as e:
                    st.error(f"TFLite conversion failed: {e}")

        # Save history entry
        history_entry = {
            "timestamp": int(time.time()),
            "num_classes": len(class_names),
            "num_images": len(X),
            "accuracy": {"logistic": float(acc_lr), "rf": float(acc_rf), "cnn": float(acc_cnn)},
            "classes": class_names
        }
        save_history_entry(history_entry)

# ---------------- Prediction & Explainability ----------------
st.header("🔮 Test Prediction & Explainability")

# Load models if present
model_files_present = os.path.exists(MODELS_DIR) and os.listdir(MODELS_DIR)
loaded_models = {}
try:
    if os.path.exists(os.path.join(MODELS_DIR, "logistic.pkl")):
        loaded_models["logistic"] = joblib.load(os.path.join(MODELS_DIR, "logistic.pkl"))
    if os.path.exists(os.path.join(MODELS_DIR, "random_forest.pkl")):
        loaded_models["rf"] = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    if os.path.exists(os.path.join(MODELS_DIR, "cnn.h5")) and TF_AVAILABLE:
        loaded_models["cnn"] = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn.h5"))
except Exception as e:
    st.warning(f"Problem loading models: {e}")

upload_test = st.file_uploader("Upload image for prediction", type=["jpg","png","jpeg"])
test_cam = st.checkbox("Or take photo with camera for test")
test_img = None
if test_cam:
    cam = st.camera_input("Take a test photo")
    if cam:
        test_img = Image.open(cam).convert("RGB")
if upload_test:
    test_img = Image.open(upload_test).convert("RGB")

if test_img is not None:
    st.image(test_img, caption="Test Image", width=250)
    img_resized = test_img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype("float32") / 255.0

    if not class_names:
        _, _, class_names = load_dataset()
    # predictions
    probs = {}
    preds = {}
    if "logistic" in loaded_models:
        try:
            p = loaded_models["logistic"].predict_proba(arr.reshape(1,-1))[0]
            probs["logistic"] = p
            preds["logistic"] = int(np.argmax(p))
        except Exception as e:
            st.warning(f"Logistic prediction failed: {e}")
    if "rf" in loaded_models:
        try:
            p = loaded_models["rf"].predict_proba(arr.reshape(1,-1))[0]
            probs["rf"] = p
            preds["rf"] = int(np.argmax(p))
        except Exception as e:
            st.warning(f"RF prediction failed: {e}")
    if "cnn" in loaded_models and TF_AVAILABLE:
        try:
            p = loaded_models["cnn"].predict(arr.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3))[0]
            probs["cnn"] = p
            preds["cnn"] = int(np.argmax(p))
        except Exception as e:
            st.warning(f"CNN prediction failed: {e}")

    st.subheader("Model Outputs")
    for k, v in preds.items():
        st.write(f"**{k.upper()}** → {class_names[v]} (confidence {probs[k][v]*100:.2f}%)")

    # Ensemble average
    if probs:
        # Align shapes and average
        p_sum = None
        n = 0
        for p in probs.values():
            if p_sum is None:
                p_sum = np.array(p)
            else:
                p_sum = p_sum + np.array(p)
            n += 1
        p_avg = p_sum / n
        p_arg = int(np.argmax(p_avg))
        st.subheader("Ensemble (average probabilities)")
        df = pd.DataFrame({"class": class_names, "prob": (p_avg*100).round(2)})
        st.dataframe(df.sort_values("prob", ascending=False).reset_index(drop=True))
        st.write(f"Ensemble prediction: **{class_names[p_arg]}** ({p_avg[p_arg]*100:.2f}%)")

    # Confidence bar visuals
    st.subheader("Confidence (per model)")
    for k, p in probs.items():
        st.write(k.upper())
        df = pd.DataFrame({"class": class_names, "prob": (np.array(p)*100).round(2)})
        st.bar_chart(df.set_index("class"))

    # Grad-CAM (only for CNN)
    if "cnn" in loaded_models and TF_AVAILABLE:
        if st.button("Show Grad-CAM for CNN"):
            try:
                model = loaded_models["cnn"]
                img_tensor = arr.reshape((1, IMG_SIZE[0], IMG_SIZE[1], 3))
                pred_index = np.argmax(model.predict(img_tensor)[0])
                # find last conv layer
                last_conv_layer_name = None
                for layer in reversed(model.layers):
                    if isinstance(layer, layers.Conv2D):
                        last_conv_layer_name = layer.name
                        break
                if last_conv_layer_name is None:
                    st.warning("No Conv2D layer found for Grad-CAM.")
                else:
                    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
                    with tf.GradientTape() as tape:
                        conv_outputs, predictions = grad_model(img_tensor)
                        loss = predictions[:, pred_index]
                    grads = tape.gradient(loss, conv_outputs)
                    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
                    conv_outputs = conv_outputs[0]
                    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                    heatmap = tf.squeeze(heatmap)
                    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
                    heatmap = Image.fromarray(np.uint8(255*heatmap.numpy())).resize(IMG_SIZE)
                    heatmap = heatmap.convert("RGBA")
                    original = test_img.resize(IMG_SIZE).convert("RGBA")
                    # overlay
                    overlay = Image.new("RGBA", IMG_SIZE)
                    # tint heatmap to red
                    hm = np.array(heatmap)
                    hm_img = Image.fromarray(np.uint8(np.dstack([hm[:,:,0], np.zeros_like(hm[:,:,0]), np.zeros_like(hm[:,:,0]), hm[:,:,0]])))
                    composite = Image.blend(original, hm_img, alpha=0.4)
                    st.image(composite, caption=f"Grad-CAM (predicted: {class_names[pred_index]})", use_column_width=False)
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")

# ---------- Training history ----------
st.header("🕒 Training History")
hist = load_history()
if not hist:
    st.info("No training history yet.")
else:
    df_hist = pd.DataFrame(hist)
    # convert timestamp to readable
    df_hist["date"] = pd.to_datetime(df_hist["timestamp"], unit="s")
    st.dataframe(df_hist[["date", "num_classes", "num_images", "accuracy"]].sort_values("date", ascending=False))

    # Plot past accuracies
    try:
        fig, ax = plt.subplots()
        ax.plot(df_hist["date"], df_hist["accuracy"].apply(lambda x: x.get("cnn", 0)), label="cnn")
        ax.plot(df_hist["date"], df_hist["accuracy"].apply(lambda x: x.get("rf", 0)), label="rf")
        ax.plot(df_hist["date"], df_hist["accuracy"].apply(lambda x: x.get("logistic", 0)), label="logistic")
        ax.set_ylabel("accuracy")
        ax.legend()
        st.pyplot(fig)
    except Exception:
        pass

st.caption("Single-file app by ChatGPT — modify features as needed. If you want ONNX export, extended duplicate detection using perceptual hashing, or removal of heavy warnings, tell me and I'll add it.")
