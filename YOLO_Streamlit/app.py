import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load model once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # 🔁 Make sure your .pt file is named like this
    return model

model = load_model()

st.title("YOLO Target Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO
    results = model(image)

    # Show detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"**{label}** with confidence {conf:.2f}")

        st.image(r.plot(), caption="Detected Targets", use_column_width=True)
