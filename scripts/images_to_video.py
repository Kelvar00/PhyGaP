import argparse
from pathlib import Path
from typing import List, Tuple

import cv2


def collect_images(input_dir: Path, recursive: bool) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)


def read_frame_size(image_path: Path) -> Tuple[int, int]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")
    height, width = frame.shape[:2]
    return width, height


def images_to_video(
    input_dir: Path,
    output_path: Path,
    fps: float,
    codec: str,
    recursive: bool,
) -> None:
    image_files = collect_images(input_dir, recursive)
    if not image_files:
        raise ValueError(f"No image files found in: {input_dir}")

    width, height = read_frame_size(image_files[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open video writer. Output: {output_path}, codec: {codec}"
        )

    written = 0
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Skip unreadable image: {img_path}")
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        writer.write(frame)
        written += 1

    writer.release()

    if written == 0:
        raise RuntimeError("No frames were written to output video.")

    print(f"[OK] Wrote {written} frames to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all images in a folder into a video (one image per frame)."
    )
    parser.add_argument("input_dir", type=Path, help="Directory that contains images")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output video path, default: <input_dir_name>.mp4 next to input_dir",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second of output video",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec, e.g. mp4v, XVID, MJPG",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subfolders recursively",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    if args.output is None:
        output_path = input_dir.parent / f"{input_dir.name}.mp4"
    else:
        output_path = args.output

    images_to_video(
        input_dir=input_dir,
        output_path=output_path,
        fps=args.fps,
        codec=args.codec,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
