import argparse
import shutil
import zipfile
from pathlib import Path

import yaml
from kaggle.api.kaggle_api_extended import KaggleApi


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_dataset(api: KaggleApi, slug: str, target_dir: Path, force: bool) -> None:
    ensure_dir(target_dir)
    if force and target_dir.exists():
        shutil.rmtree(target_dir)
        ensure_dir(target_dir)

    zip_path = target_dir / f"{slug.replace('/', '_')}.zip"
    api.dataset_download_files(slug, path=str(target_dir), unzip=False, quiet=False)

    downloaded_archives = sorted(target_dir.glob("*.zip"), key=lambda item: item.stat().st_mtime, reverse=True)
    if downloaded_archives:
        archive = downloaded_archives[0]
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir)
        if archive != zip_path and zip_path.exists():
            zip_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle_datasets.yaml")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    selected = set(args.only)

    api = KaggleApi()
    api.authenticate()

    for dataset in config.get("datasets", []):
        slug = dataset["slug"]
        if selected and slug not in selected:
            continue
        target_dir = Path(dataset["target_dir"])
        print(f"Downloading {slug} -> {target_dir}")
        download_dataset(api, slug, target_dir, args.force)


if __name__ == "__main__":
    main()
