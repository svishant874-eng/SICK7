import os
import shutil
# pyrefly: ignore [missing-import]
from PIL import Image


# ==========================================
# PATHS
# ==========================================

TRAIN_IMAGES = r"C:\Users\MANJEET\Downloads\VisDrone2019-DET-train\VisDrone2019-DET-train\images"
TRAIN_ANNOTATIONS = r"C:\Users\MANJEET\Downloads\VisDrone2019-DET-train\VisDrone2019-DET-train\annotations"

VAL_IMAGES = r"C:\Users\MANJEET\Downloads\VisDrone2019-DET-val\VisDrone2019-DET-val\images"
VAL_ANNOTATIONS = r"C:\Users\MANJEET\Downloads\VisDrone2019-DET-val\VisDrone2019-DET-val\annotations"

OUTPUT = r"C:\Users\MANJEET\Desktop\SICK7\SIH\person_dataset"


# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs(os.path.join(OUTPUT, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "images", "val"), exist_ok=True)

os.makedirs(os.path.join(OUTPUT, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "labels", "val"), exist_ok=True)


# ==========================================
# CONVERT VISDRONE → YOLO
# ==========================================

def convert_dataset(image_dir, annotation_dir, split):

    output_images = os.path.join(
        OUTPUT, "images", split
    )

    output_labels = os.path.join(
        OUTPUT, "labels", split
    )

    converted_images = 0
    total_persons = 0

    print(f"\nConverting {split} dataset...")

    for annotation_file in os.listdir(annotation_dir):

        if not annotation_file.endswith(".txt"):
            continue

        annotation_path = os.path.join(
            annotation_dir,
            annotation_file
        )

        # ------------------------------------------
        # Corresponding image
        # ------------------------------------------

        image_name = os.path.splitext(annotation_file)[0] + ".jpg"

        image_path = os.path.join(
            image_dir,
            image_name
        )

        if not os.path.exists(image_path):
            continue

        # ------------------------------------------
        # Read image dimensions
        # ------------------------------------------

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size

        except Exception as e:
            print(f"Could not read {image_name}: {e}")
            continue

        person_labels = []

        # ------------------------------------------
        # Read VisDrone annotations
        #
        # VisDrone format:
        # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
        # <score>,<object_category>,<truncation>,<occlusion>
        # ------------------------------------------

        with open(annotation_path, "r") as file:

            lines = file.readlines()

        for line in lines:

            parts = line.strip().split(",")

            if len(parts) < 8:
                continue

            try:

                x = float(parts[0])
                y = float(parts[1])
                width = float(parts[2])
                height = float(parts[3])

                score = int(parts[4])
                category = int(parts[5])

            except ValueError:
                continue

            # --------------------------------------
            # In VisDrone ground truth:
            # score == 0 indicates ignored regions
            # --------------------------------------

            if score == 0:
                continue

            # --------------------------------------
            # Person classes:
            # 1 = pedestrian
            # 2 = people
            # --------------------------------------

            if category not in [1, 2]:
                continue

            # --------------------------------------
            # Clip bounding boxes to image boundaries
            # --------------------------------------

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(image_width), x + width)
            y2 = min(float(image_height), y + height)

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            # --------------------------------------
            # Convert to YOLO format
            #
            # YOLO:
            # class x_center y_center width height
            # --------------------------------------

            x_center = (x1 + w / 2.0) / image_width
            y_center = (y1 + h / 2.0) / image_height

            normalized_width = w / image_width
            normalized_height = h / image_height

            # --------------------------------------
            # Class 0 = person
            # --------------------------------------

            person_labels.append(
                f"0 {x_center:.6f} "
                f"{y_center:.6f} "
                f"{normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )

        # ------------------------------------------
        # Only copy images containing persons
        # ------------------------------------------

        if len(person_labels) > 0:

            shutil.copy2(
                image_path,
                os.path.join(
                    output_images,
                    image_name
                )
            )

            label_path = os.path.join(
                output_labels,
                annotation_file
            )

            with open(label_path, "w") as file:

                file.write(
                    "\n".join(person_labels)
                )

            converted_images += 1
            total_persons += len(person_labels)

    print(
        f"{split}: "
        f"{converted_images} images converted, "
        f"{total_persons} persons found"
    )


# ==========================================
# CONVERT TRAIN + VAL
# ==========================================

print("==========================================")
print("     SICK7 PERSON DATASET CONVERTER")
print("==========================================")

convert_dataset(
    TRAIN_IMAGES,
    TRAIN_ANNOTATIONS,
    "train"
)

convert_dataset(
    VAL_IMAGES,
    VAL_ANNOTATIONS,
    "val"
)


# ==========================================
# CREATE data.yaml
# ==========================================

yaml_content = """path: ./person_dataset

train: images/train
val: images/val

names:
  0: person
"""

yaml_path = os.path.join(
    OUTPUT,
    "data.yaml"
)

with open(yaml_path, "w") as file:
    file.write(yaml_content)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("Person dataset conversion completed!")
print("==========================================")

print(f"\nDataset location:")
print(OUTPUT)

print("\nClasses:")
print("0: person")

print("\nDataset structure:")
print("person_dataset/")
print("├── images/")
print("│   ├── train/")
print("│   └── val/")
print("├── labels/")
print("│   ├── train/")
print("│   └── val/")
print("└── data.yaml")