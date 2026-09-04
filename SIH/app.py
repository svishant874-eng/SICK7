# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import tempfile
import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image
from ultralytics import YOLO
import re
from pathlib import Path

MODEL_CHOICES = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8m-seg",
    "yolov8n-pose",
    "yolov8s-pose",
    "yolov8m-pose",
    "yolov8n-obb",
    "yolov8s-obb",
    "yolov8m-obb",
    "yolov8n-cls",
    "yolov8s-cls",
    "yolov8m-cls",
]

IMAGE_SIZE_CHOICES = [320, 640, 1024]
CUSTOM_CSS = (Path(__file__).parent / "ultralytics.css").read_text()

# Cache OCR reader for performance
_ocr_reader = None

def get_ocr_reader():
    """Initializes and returns cached EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

INDIAN_STATES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

CHAR_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'B': '8', 'S': '5', 'Z': '2', 'I': '1', 'A': '4', 'G': '6'}
DIGIT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '6': 'G'}

def clean_plate_text(raw_text):
    """Clean and standardize plate text to alphanumeric characters."""
    return re.sub(r'[^A-Z0-9]', '', raw_text.upper().strip())

def format_and_validate_plate(text):
    """
    Validates and formats license plate string.
    Checks for Indian format: [State 2L][District 2D][Optional 1-2L][Number 4D]
    Or generic international format: 4 to 10 alphanumeric characters.
    """
    clean = clean_plate_text(text)
    if len(clean) < 4 or len(clean) > 12:
        return None, 0.0

    score = 0.5

    # Try Indian plate correction
    if len(clean) >= 8 and len(clean) <= 11:
        state_part = list(clean[:2])
        for i in range(2):
            if state_part[i] in DIGIT_TO_CHAR:
                state_part[i] = DIGIT_TO_CHAR[state_part[i]]
        state_str = "".join(state_part)

        if state_str in INDIAN_STATES:
            score += 0.4
            dist_part = list(clean[2:4])
            for i in range(2):
                if dist_part[i] in CHAR_TO_DIGIT:
                    dist_part[i] = CHAR_TO_DIGIT[dist_part[i]]
            dist_str = "".join(dist_part)

            num_part = list(clean[-4:])
            for i in range(len(num_part)):
                if num_part[i] in CHAR_TO_DIGIT:
                    num_part[i] = CHAR_TO_DIGIT[num_part[i]]
            num_str = "".join(num_part)

            mid_str = clean[4:-4]
            mid_part = [DIGIT_TO_CHAR.get(c, c) for c in mid_str]
            mid_str = "".join(mid_part)

            formatted = f"{state_str} {dist_str}"
            if mid_str:
                formatted += f" {mid_str}"
            formatted += f" {num_str}"
            return formatted, score

    # Generic alphanumeric plate validation
    has_letters = any(c.isalpha() for c in clean)
    has_digits = any(c.isdigit() for c in clean)

    if has_letters and has_digits:
        score += 0.3
        formatted = re.sub(r'([A-Z]+)', r' \1 ', clean).strip()
        formatted = re.sub(r'\s+', ' ', formatted)
        return formatted, score
    elif len(clean) >= 5 and score >= 0.5:
        return clean, score

    return None, 0.0

def find_license_plate_candidates(vehicle_crop):
    """
    Locates where the number plate is likely present on a vehicle:
    1. Bumper ROI prior (plates are in the bottom 50% and horizontally centered).
    2. Vertical edge filtering (Sobel-X) + morphological closing to extract plate-shaped rectangles.
    """
    h, w, _ = vehicle_crop.shape
    candidates = []

    # 1. Primary Bumper ROI: lower 55% of the vehicle, horizontal center 80%
    y_start = int(h * 0.45)
    y_end = int(h * 0.96)
    x_start = int(w * 0.10)
    x_end = int(w * 0.90)

    bumper = vehicle_crop[y_start:y_end, x_start:x_end]
    if bumper.size == 0:
        return [vehicle_crop]

    candidates.append(bumper)

    # 2. Morphological Edge Detection on Bumper
    bh, bw, _ = bumper.shape
    gray = cv2.cvtColor(bumper, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0:
            continue
        aspect_ratio = float(cw) / ch
        area = cw * ch
        if 1.8 <= aspect_ratio <= 6.0 and area > (bw * bh * 0.01) and area < (bw * bh * 0.7):
            pad_x = int(cw * 0.15)
            pad_y = int(ch * 0.20)
            cx1 = max(0, x - pad_x)
            cy1 = max(0, y - pad_y)
            cx2 = min(bw, x + cw + pad_x)
            cy2 = min(bh, y + ch + pad_y)
            crop = bumper[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                contour_candidates.append((area, crop))

    contour_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, c_crop in contour_candidates[:3]:
        candidates.append(c_crop)

    candidates.append(vehicle_crop)
    return candidates

def preprocess_plate_for_ocr(crop):
    """Enhance plate contrast and resolution for OCR accuracy."""
    h, w = crop.shape[:2]
    if h < 80 or w < 240:
        scale = max(2.0, 100.0 / max(h, 1))
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return crop, enhanced

def extract_number_plate(crop_bgr):
    """
    Localizes plate region on vehicle, enhances contrast, reads characters with EasyOCR,
    and applies standard plate validation. Returns recognized number or 'Not Visible'.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "Not Visible"

    try:
        reader = get_ocr_reader()
        candidates = find_license_plate_candidates(crop_bgr)
        best_plate = None
        best_score = 0.0
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -'

        for candidate_crop in candidates:
            color_crop, enhanced = preprocess_plate_for_ocr(candidate_crop)

            for img_variant in [enhanced, color_crop]:
                try:
                    ocr_results = reader.readtext(
                        img_variant,
                        allowlist=allowlist,
                        detail=1,
                        paragraph=False,
                        width_ths=0.7
                    )

                    texts = [r[1] for r in ocr_results if r[2] > 0.25]
                    combined = " ".join(texts)

                    formatted, score = format_and_validate_plate(combined)
                    if formatted and score > best_score:
                        best_plate = formatted
                        best_score = score
                        if score >= 0.8:
                            return best_plate

                    for r in ocr_results:
                        formatted_tok, tok_score = format_and_validate_plate(r[1])
                        if formatted_tok and (tok_score + r[2] * 0.2) > best_score:
                            best_plate = formatted_tok
                            best_score = tok_score + r[2] * 0.2
                except Exception:
                    pass

        if best_plate and best_score >= 0.6:
            return best_plate

    except Exception as e:
        print(f"License plate extraction error: {e}")

    return "Not Visible"

COLORS = [
    (0, 230, 115),   # Emerald Green
    (255, 140, 0),   # Orange
    (230, 40, 100),  # Pink/Red
    (40, 120, 255),  # Sky Blue
    (200, 50, 255),  # Purple
    (255, 215, 0),   # Gold
    (0, 210, 255),   # Cyan
    (255, 99, 71),   # Tomato
]

def predict_image(img, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz, assign_ids=True, detect_plate=True):
    """Predicts objects with unique ID assignment, license plate recognition, cropped gallery, and details table."""
    if img is None:
        return None, [], []

    model = get_model(model_name)
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    r = results[0]
    boxes = r.boxes

    if not assign_ids or len(boxes) == 0:
        im_array = r.plot(labels=show_labels, conf=show_conf)
        im = Image.fromarray(im_array[..., ::-1])
        return im, [], []

    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    scale = 2 if w < 600 else 1
    if scale > 1:
        annotated = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    else:
        annotated = img_bgr.copy()

    gallery_crops = []
    table_rows = []

    for idx, box in enumerate(boxes):
        obj_id = idx + 1
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w, x2), min(h, y2)

        crop_bgr = img_bgr[y1_c:y2_c, x1_c:x2_c]
        is_vehicle = cls_name.lower() in VEHICLE_CLASSES
        plate_text = "N/A"
        if is_vehicle and detect_plate:
            plate_text = extract_number_plate(crop_bgr)

        caption = f"ID #{obj_id}: {cls_name.title()} ({int(conf * 100)}%)"
        if is_vehicle and detect_plate:
            caption += f" • Plate: {plate_text}"

        if crop_bgr.size > 0:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            gallery_crops.append((crop_rgb, caption))

        table_rows.append([f"#{obj_id}", cls_name.title(), f"{conf:.2%}", plate_text if detect_plate else "N/A", f"[{x1}, {y1}, {x2}, {y2}]"])

        color = COLORS[(obj_id - 1) % len(COLORS)]
        sx1, sy1, sx2, sy2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), color, 2)

        tag_text = f"#{obj_id} {cls_name.upper()}"
        if show_conf:
            tag_text += f" {int(conf * 100)}%"
        if is_vehicle and detect_plate:
            tag_text += f" | Plate: {plate_text}"

        font_scale = 0.45 if scale > 1 else max(0.4, w / 1500)
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tag_y1 = max(0, sy1 - th - 8)
        tag_y2 = sy1
        cv2.rectangle(annotated, (sx1, tag_y1), (sx1 + tw + 10, tag_y2), color, -1)
        cv2.putText(annotated, tag_text, (sx1 + 5, sy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), gallery_crops, table_rows


def predict_video(video_path, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in a video using a Ultralytics YOLO model and returns the annotated video."""
    if video_path is None:
        return None

    model = YOLO(model_name)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_output.name
    temp_output.close()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        # Get the annotated frame
        annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
        out.write(annotated_frame)

    cap.release()
    out.release()

    return output_path

# Cache model for streaming performance
_model_cache = {}

def get_model(model_name):
    """Get or create a cached model instance."""
    if model_name not in _model_cache:
        _model_cache[model_name] = YOLO(model_name)
    return _model_cache[model_name]


def predict_webcam(frame, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in a webcam frame using a Ultralytics YOLO model (optimized for streaming)."""
    if frame is None:
        return None

    # Use cached model for better streaming performance
    model = get_model(model_name)

    if isinstance(frame, np.ndarray):
        # Gradio webcam sends RGB, but Ultralytics YOLO expects BGR for OpenCV operations
        # Convert RGB to BGR for YOLO
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = model.predict(
            source=frame_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        # YOLO's plot() returns BGR, convert back to RGB for Gradio display
        annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
        # Convert BGR to RGB for Gradio
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    return None


# Create the Gradio app with tabs
with gr.Blocks(title="Ultralytics YOLOv8 Inference 🚀") as demo:
    gr.Markdown("# Ultralytics YOLOv8 Inference 🚀")
    gr.Markdown("Upload images, videos, or use your webcam for real-time detection, segmentation, pose estimation, OBB, and classification.")

    with gr.Tabs():
        # Image Tab
        with gr.TabItem("📷 Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    img_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    img_model = gr.Radio(choices=MODEL_CHOICES, label="Model Name", value="yolov8n")
                    img_assign_id = gr.Checkbox(value=True, label="Assign Unique Object IDs & Crop Each Object")
                    img_detect_plate = gr.Checkbox(value=True, label="Detect Number Plates on Vehicles (OCR)")
                    img_labels = gr.Checkbox(value=True, label="Show Labels")
                    img_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    img_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                    img_btn = gr.Button("Detect & Identify Objects", variant="primary")
                with gr.Column(scale=1):
                    img_output = gr.Image(type="pil", label="Annotated Result with Assigned IDs & Plates")
                    img_table = gr.Dataframe(headers=["ID", "Class", "Confidence", "Number Plate", "Coordinates [x1, y1, x2, y2]"], label="Detection & Number Plate Details")

            gr.Markdown("### 🔍 Individual Detected Objects / Persons (with Assigned IDs & Plates)")
            img_gallery = gr.Gallery(label="Detected Object Crops", columns=4, height="auto")

            img_btn.click(
                predict_image,
                inputs=[img_input, img_conf, img_iou, img_model, img_labels, img_conf_show, img_size, img_assign_id, img_detect_plate],
                outputs=[img_output, img_gallery, img_table],
            )

            gr.Examples(
                examples=[
                    ["OIP.webp", 0.35, 0.7, "yolov8m", True, True, 640, True, True],
                    ["https://ultralytics.com/images/bus.jpg", 0.25, 0.7, "yolov8n", True, True, 640, True, True],
                    ["https://ultralytics.com/images/zidane.jpg", 0.25, 0.7, "yolov8n-seg", True, True, 640, True, False],
                ],
                inputs=[img_input, img_conf, img_iou, img_model, img_labels, img_conf_show, img_size, img_assign_id, img_detect_plate],
            )

        # Video Tab
        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video")
                    vid_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    vid_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    vid_model = gr.Radio(choices=MODEL_CHOICES, label="Model Name", value="yolov8n")
                    vid_labels = gr.Checkbox(value=True, label="Show Labels")
                    vid_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    vid_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                    vid_btn = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Result")

            vid_btn.click(
                predict_video,
                inputs=[vid_input, vid_conf, vid_iou, vid_model, vid_labels, vid_conf_show, vid_size],
                outputs=vid_output,
            )

        # Webcam Tab - Real-time streaming
        with gr.TabItem("📹 Webcam"):
            gr.Markdown("### Real-time Webcam Detection")
            gr.Markdown("Enable streaming for live detection as you move!")
            with gr.Row():
                with gr.Column():
                    webcam_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    webcam_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    webcam_model = gr.Radio(choices=MODEL_CHOICES, label="Model Name", value="yolov8n")
                    webcam_labels = gr.Checkbox(value=True, label="Show Labels")
                    webcam_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    webcam_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                with gr.Column():
                    # Streaming webcam input with real-time output
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam (streaming)",
                        streaming=True,
                    )
                    webcam_output = gr.Image(type="numpy", label="Detection Result")

            # Stream event for real-time detection
            webcam_input.stream(
                predict_webcam,
                inputs=[
                    webcam_input,
                    webcam_conf,
                    webcam_iou,
                    webcam_model,
                    webcam_labels,
                    webcam_conf_show,
                    webcam_size,
                ],
                outputs=webcam_output,
            )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, ssr_mode=False)
