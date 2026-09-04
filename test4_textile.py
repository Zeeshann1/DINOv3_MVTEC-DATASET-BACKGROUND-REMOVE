import os
import pickle
import pandas as pd
from detectron2.engine import DefaultPredictor

print("✅ SAM 2 imports successful! zee")
# SAM 2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import torch
from pathlib import Path

print("✅ SAM 2 imports successful!")

def load_detectron2_model(cfg_path: str, weights_path: str, score_thresh=0.7):
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)


def process_image(image_path, output_dir, sam2_predictor, detector):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️ Skipping invalid image: {image_path}")
        return


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = detector(image_rgb)
    instances = outputs["instances"]

    # ==============================
    # STEP 1: Process TAPE (class 0)
    # ==============================
    tape_mask_cond = instances.pred_classes == 0
    if not torch.any(tape_mask_cond):
        print(f"ℹ️ No tape (class 0) found in {image_path}")
        return

    # Get indices of tape instances
    tape_indices = torch.where(tape_mask_cond)[0]
    tape_scores = instances.scores[tape_mask_cond]
    best_local_idx = tape_scores.argmax()
    best_global_idx = tape_indices[best_local_idx].item()

    # Extract box safely (keep 2D shape for Boxes)
    tape_box = instances.pred_boxes[[best_global_idx]].tensor.cpu().numpy()[0]

    # Generate SAM2 mask for tape
    sam2_predictor.set_image(image_rgb)
    x1, y1, x2, y2 = map(int, tape_box)
    center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
    input_box = np.array([x1, y1, x2, y2])

    masks, _, _ = sam2_predictor.predict(
        point_coords=center,
        point_labels=np.array([1]),
        box=input_box[None, :],
        multimask_output=False
    )

    if len(masks) == 0:
        print(f"⚠️ SAM2 failed for tape in {image_path}")
        return

    # Clean tape mask
    tape_mask = (masks[0] > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_CLOSE, kernel)
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(tape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No valid tape contour in {image_path}")
        return
    largest_contour = max(contours, key=cv2.contourArea)
    tape_mask_clean = np.zeros_like(tape_mask)
    cv2.fillPoly(tape_mask_clean, [largest_contour], 255)
    tape_mask = tape_mask_clean

    # Initialize foreground: keep only tape
    foreground = image.copy()
    foreground[tape_mask == 0] = 0

    # ==============================
    # STEP 2: Remove DEFECTS (class 1)
    # ==============================
    defect_mask_cond = instances.pred_classes == 1
    if torch.any(defect_mask_cond):
        defect_boxes = instances.pred_boxes[defect_mask_cond].tensor.cpu().numpy()

        sam2_predictor.set_image(image_rgb)  # Re-embed for defect refinement

        for box in defect_boxes:
            x1, y1, x2, y2 = map(int, box)
            center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            input_box = np.array([x1, y1, x2, y2])

            masks_def, _, _ = sam2_predictor.predict(
                point_coords=center,
                point_labels=np.array([1]),
                box=input_box[None, :],
                multimask_output=False
            )

            if len(masks_def) == 0:
                continue

            defect_mask = (masks_def[0] > 0.5).astype(np.uint8) * 255

            # Clean defect mask
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel_close)

            # Fill holes
            contours_d, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_d:
                largest_d = max(contours_d, key=cv2.contourArea)
                filled_d = np.zeros_like(defect_mask)
                cv2.fillPoly(filled_d, [largest_d], 255)
                defect_mask = filled_d

            # Optional: slight dilation for safety
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            defect_mask = cv2.dilate(defect_mask, kernel_dilate, iterations=1)

            # Only erase defects that lie within the tape region
            defect_in_tape = cv2.bitwise_and(defect_mask, tape_mask)
            foreground[defect_in_tape > 0] = 0


    # ==============================
    # STEP 3: Center in 448x448
    # ==============================
    y_idx, x_idx = np.where(tape_mask > 0)
    if len(y_idx) == 0 or len(x_idx) == 0:
        print(f"⚠️ No foreground after defect removal in {image_path}")
        return

    y_min, y_max = y_idx.min(), y_idx.max()
    x_min, x_max = x_idx.min(), x_idx.max()

    cropped_fg = foreground[y_min:y_max + 1, x_min:x_max + 1]


    # Target size
    target_size = 1280
    # Get original dimensions
    h, w = cropped_fg.shape[:2]
    # Compute scaling factor to fit within 448x448 while preserving aspect ratio
    scale = min(target_size / w, target_size / h)
    # Resize image
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(cropped_fg, (new_w, new_h))
    # Create a 448x448 blank canvas (filled with white or any color)
    canvas = np.full((target_size, target_size, 3), 0, dtype=np.uint8) # 0 = black padding, 255 for white
    # Compute top-left corner to center the resized image
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    # Place resized image on canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    # Now `canvas` is your 448x448 image with preserved content and padding


    # STEP 4: Save
    output_path = output_dir / f"{image_path.stem}_centered448{image_path.suffix}"
    cv2.imwrite(str(output_path), canvas)  #output_img

    return {"success": True}


def main():
    current_dir = Path.cwd()
    main_folder = current_dir.parent

    cfg_path = "weights/A350_DUT/v1/OD_CFG.pickle"
    weights_path = "weights/A350_DUT/v1/model_final.pth"
    detector = load_detectron2_model(cfg_path, weights_path)

    #sam2_checkpoint = main_folder / "sam2/checkpoints/sam2.1_hiera_tiny.pt"
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, str(sam2_checkpoint), device="cpu")
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    input_path = Path("data/A350_defect/v2")
    output_dir = Path("mvtec_result/A350_defect/v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".HEIC"}
        ])

    for img_path in image_paths:
        print("Processing Image: ", img_path)
        process_image(img_path, output_dir, sam2_predictor, detector)


if __name__ == "__main__":
    print("Starting textile tape analysis with 360° angle and clockwise rotation to 90°...")
    main()