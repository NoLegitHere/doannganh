import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_CLASS_ALIASES = {"license_plate", "plate", "number_plate", "licence_plate", "vehicle_registration_plate"}


def ensure_structure(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def find_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return files


def parse_voc_annotation(xml_path: Path) -> list[tuple[str, int, int, int, int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations: list[tuple[str, int, int, int, int]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = int(float(bbox.findtext("xmin", "0")))
        ymin = int(float(bbox.findtext("ymin", "0")))
        xmax = int(float(bbox.findtext("xmax", "0")))
        ymax = int(float(bbox.findtext("ymax", "0")))
        annotations.append((name, xmin, ymin, xmax, ymax))
    return annotations


def voc_to_yolo_line(label: tuple[str, int, int, int, int], width: int, height: int) -> str | None:
    name, xmin, ymin, xmax, ymax = label
    if name not in PLATE_CLASS_ALIASES:
        return None
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def resolve_label_for_image(image_path: Path) -> tuple[str, Path] | None:
    txt_label = image_path.with_suffix(".txt")
    xml_label = image_path.with_suffix(".xml")

    if txt_label.exists():
        return "yolo", txt_label
    if xml_label.exists():
        return "voc", xml_label

    for sibling in image_path.parent.glob(f"{image_path.stem}.*"):
        if sibling.suffix.lower() == ".txt":
            return "yolo", sibling
        if sibling.suffix.lower() == ".xml":
            return "voc", sibling
    return None


def choose_split(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    threshold_train = int(total * train_ratio)
    threshold_val = int(total * (train_ratio + val_ratio))
    if index < threshold_train:
        return "train"
    if index < threshold_val:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="data/processed/detection")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = Path(args.output)
    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_structure(output_root)

    image_paths: list[Path] = []
    for input_dir in args.inputs:
        image_paths.extend(find_images(Path(input_dir)))

    random.seed(args.seed)
    random.shuffle(image_paths)

    kept = 0
    skipped = 0

    for index, image_path in enumerate(image_paths):
        label_info = resolve_label_for_image(image_path)
        if label_info is None:
            skipped += 1
            continue

        split = choose_split(index, len(image_paths), args.train_ratio, args.val_ratio)
        target_image = output_root / "images" / split / image_path.name
        target_label = output_root / "labels" / split / f"{image_path.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)

        label_type, label_path = label_info
        if label_type == "yolo":
            shutil.copy2(label_path, target_label)
        else:
            import cv2

            image = cv2.imread(str(image_path))
            if image is None:
                skipped += 1
                continue
            height, width = image.shape[:2]
            lines = []
            for annotation in parse_voc_annotation(label_path):
                line = voc_to_yolo_line(annotation, width, height)
                if line is not None:
                    lines.append(line)
            if not lines:
                target_image.unlink(missing_ok=True)
                skipped += 1
                continue
            target_label.write_text("\n".join(lines), encoding="utf-8")

        kept += 1

    print(f"Prepared {kept} labeled images. Skipped {skipped} unlabeled/invalid images.")
    print(f"Output dataset: {output_root}")


if __name__ == "__main__":
    main()
