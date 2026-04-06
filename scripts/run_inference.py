import argparse
from pathlib import Path

import cv2

from app.services.image_utils import read_image_from_path
from app.services.recognition_pipeline import PlateRecognitionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", default="outputs/inference/result.jpg")
    args = parser.parse_args()

    image = read_image_from_path(args.image)
    pipeline = PlateRecognitionPipeline()
    response = pipeline.recognize(image)
    annotated = pipeline.annotate(image, response)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    print(response.model_dump_json(indent=2, ensure_ascii=False))
    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
