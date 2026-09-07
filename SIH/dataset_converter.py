import os
import shutil
from PIL import Image

# ==========================================
# 1. VISDRONE DATASET PATHS
# ==========================================

TRAIN_PATH = r"C:\Users\MANJEET\Downloads\VisDrone2019-DET-train\VisDrone2019-DET-train"
VAL_PATH = r"C:\Users\MANJEET\Downloads\VisDrone2019-DET-val\VisDrone2019-DET-val"

# Output dataset
OUTPUT_PATH = r"C:\Users\MANJEET\Desktop\SICK7\SIH\vehicle_dataset"


# ==========================================
# 2. VISDRONE CLASS MAPPING
# ==========================================

# VisDrone classes:
# 4  = car
# 5  = van
# 6  = truck
# 9  = bus
# 10 = motor

# Our classes:
# 0 = car
# 1 = truck
# 2 = bus
# 3 = motorcycle

CLASS_MAPPING = {
    4: 0,    # car -> car
    5: 0,    # van -> car
    6: 1,    # truck -> truck
    9: 2,    # bus -> bus
    10: 3    # motor -> motorcycle
}


# ==========================================
# 3. CONVERSION FUNCTION
# ==========================================

def convert_dataset(source_path, split):

    image_source = os.path.join(source_path, "images")
    label_source = os.path.join(source_path, "annotations")

    image_output = os.path.join(
        OUTPUT_PATH, "images", split
    )

    label_output = os.path.join(
        OUTPUT_PATH, "labels", split
    )

    os.makedirs(image_output, exist_ok=True)
    os.makedirs(label_output, exist_ok=True)

    annotation_files = os.listdir(label_source)

    converted = 0

    print(f"\nProcessing {split} dataset...")

    for annotation_file in annotation_files:

        if not annotation_file.endswith(".txt"):
            continue

        annotation_path = os.path.join(
            label_source,
            annotation_file
        )

        # Corresponding image
        image_name = annotation_file.replace(
            ".txt",
            ".jpg"
        )

        image_path = os.path.join(
            image_source,
            image_name
        )

        if not os.path.exists(image_path):
            continue

        # Get image size
        image = Image.open(image_path)

        image_width, image_height = image.size

        yolo_labels = []

        with open(
            annotation_path,
            "r"
        ) as file:

            for line in file:

                values = line.strip().split(",")

                if len(values) < 8:
                    continue

                # VisDrone format:
                # x, y, width, height,
                # score, class, truncation, occlusion

                class_id = int(values[5])

                # Ignore non-vehicle objects
                if class_id not in CLASS_MAPPING:
                    continue

                x = float(values[0])
                y = float(values[1])
                width = float(values[2])
                height = float(values[3])

                # Convert to YOLO format
                x_center = (
                    x + width / 2
                ) / image_width

                y_center = (
                    y + height / 2
                ) / image_height

                width = width / image_width
                height = height / image_height

                new_class = CLASS_MAPPING[class_id]

                label = (
                    f"{new_class} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

                yolo_labels.append(label)

        # If image has no vehicle, skip it
        if len(yolo_labels) == 0:
            continue

        # Copy image
        shutil.copy2(
            image_path,
            os.path.join(
                image_output,
                image_name
            )
        )

        # Save YOLO label
        output_label = os.path.join(
            label_output,
            annotation_file
        )

        with open(
            output_label,
            "w"
        ) as file:

            file.write(
                "\n".join(yolo_labels)
            )

        converted += 1

    print(
        f"{split} conversion complete!"
    )

    print(
        f"Vehicle images: {converted}"
    )


# ==========================================
# 4. START CONVERSION
# ==========================================

print(
    "======================================"
)

print(
    "Starting VisDrone conversion..."
)

print(
    "======================================"
)

convert_dataset(
    TRAIN_PATH,
    "train"
)

convert_dataset(
    VAL_PATH,
    "val"
)

print(
    "\n======================================"
)

print(
    "DATASET CONVERSION COMPLETE!"
)

print(
    "======================================"
)

print(
    f"Dataset saved at:\n{OUTPUT_PATH}"
)