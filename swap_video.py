from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, cast
from urllib.request import urlretrieve

import cv2
import imageio_ffmpeg
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm


DEFAULT_SOURCE = Path("images/pic-race.png")
DEFAULT_TARGET = Path("videos/01-1-70.mp4")
DEFAULT_OUTPUT = Path("videos/01-1-70-swapped.mp4")
MODEL_URL = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "inswapper_128_fp16.onnx"
ANALYSIS_MODEL_ROOT = Path(".insightface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap a face from an image into a video.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source face image path.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Target video path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output video path.")
    parser.add_argument(
        "--temp-output",
        type=Path,
        default=Path("videos/.tmp-01-1-70-swapped.mp4"),
        help="Temporary video path before audio is muxed back.",
    )
    parser.add_argument(
        "--execution-provider",
        action="append",
        dest="execution_providers",
        default=None,
        help="ONNX Runtime execution provider. Repeat to pass multiple providers.",
    )
    parser.add_argument(
        "--swap-all-faces",
        action="store_true",
        help="Swap all detected faces in each frame instead of only the largest face.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N frames for testing. 0 means full video.",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 640),
        help="Face detector input size.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary video without remuxing cleanup.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_swapper_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    print(f"Downloading face swap model to {model_path} ...")
    urlretrieve(MODEL_URL, model_path)
    return model_path


def build_face_analyser(model_root: Path, det_size: Sequence[int], providers: Sequence[str]) -> FaceAnalysis:
    analyser = FaceAnalysis(name="buffalo_l", root=str(model_root), providers=list(providers))
    analyser.prepare(ctx_id=0, det_size=tuple(det_size))
    return analyser


def choose_target_faces(faces: Iterable[Any], swap_all_faces: bool) -> List[Any]:
    faces = list(faces)
    if not faces:
        return []
    faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
    return faces if swap_all_faces else faces[:1]


def load_source_face(image_path: Path, analyser: FaceAnalysis) -> Any:
    source_image = cv2.imread(str(image_path))
    if source_image is None:
        raise ValueError(f"Unable to read source image: {image_path}")

    faces = analyser.get(source_image)
    if not faces:
        raise ValueError(f"No face detected in source image: {image_path}")

    faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
    return faces[0]


def get_video_properties(capture: cv2.VideoCapture) -> tuple[float, int, int, int]:
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frame_count


def create_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_factory = cast(Any, getattr(cv2, "VideoWriter_fourcc"))
    fourcc = fourcc_factory(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video for writing: {output_path}")
    return writer


def remux_audio(original_video: Path, silent_video: Path, final_video: Path) -> bool:
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not ffmpeg_exe.exists():
        return False

    final_video.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(ffmpeg_exe),
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(original_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(final_video),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0:
        return True

    print("Audio remux failed, keeping silent output.")
    print(completed.stderr.strip())
    return False


def process_video(args: argparse.Namespace) -> Path:
    ensure_exists(args.source, "Source image")
    ensure_exists(args.target, "Target video")

    providers = args.execution_providers or ["CPUExecutionProvider"]
    ensure_swapper_model(MODEL_PATH)
    analyser = build_face_analyser(ANALYSIS_MODEL_ROOT, args.det_size, providers)
    swapper = cast(Any, insightface.model_zoo.get_model(str(MODEL_PATH), providers=providers))
    source_face = load_source_face(args.source, analyser)

    capture = cv2.VideoCapture(str(args.target))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open target video: {args.target}")

    fps, width, height, total_frames = get_video_properties(capture)
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    writer = create_writer(args.temp_output, fps, (width, height))
    processed_frames = 0

    try:
        with tqdm(total=total_frames or None, unit="frame", desc="Swapping") as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if args.max_frames > 0 and processed_frames >= args.max_frames:
                    break

                target_faces = choose_target_faces(analyser.get(frame), args.swap_all_faces)
                result_frame = cast(np.ndarray, frame)
                for target_face in target_faces:
                    result_frame = cast(np.ndarray, swapper.get(result_frame, target_face, source_face, paste_back=True))

                writer.write(result_frame)
                processed_frames += 1
                progress.update(1)
    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        raise RuntimeError("No frames were processed.")

    if remux_audio(args.target, args.temp_output, args.output):
        if not args.keep_temp and args.temp_output.exists():
            args.temp_output.unlink()
        return args.output

    shutil.copyfile(args.temp_output, args.output)
    if not args.keep_temp and args.temp_output.exists():
        args.temp_output.unlink()
    return args.output


def main() -> int:
    args = parse_args()
    try:
        output_path = process_video(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Output written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())