from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, cast
from urllib.request import urlretrieve

import cv2
import imageio_ffmpeg
import insightface
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from tqdm import tqdm


DEFAULT_SOURCE = Path("images/pic-race.png")
DEFAULT_TARGET = Path("videos/01-1-70.mp4")
DEFAULT_OUTPUT = Path("videos/01-1-70-swapped.mp4")
MODEL_URL = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "inswapper_128_fp16.onnx"
ANALYSIS_MODEL_ROOT = Path(".insightface")


def _load_kv_config(config_path: Path) -> dict[str, str]:
    raise RuntimeError("_load_kv_config was replaced by _load_kv_config_jobs")


def _load_kv_config_jobs(config_path: Path) -> Tuple[dict[str, str], List[dict[str, str]]]:
    if not config_path.exists():
        return {}, []

    def parse_block(lines: List[str]) -> Tuple[dict[str, str], dict[str, List[str]]]:
        config: dict[str, str] = {}
        values_by_key: dict[str, List[str]] = {}
        current_key: Optional[str] = None
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("#", ";", "//")):
                continue

            if "=" not in line:
                if current_key is None:
                    continue
                value = line.strip().strip('"').strip("'")
                if not value:
                    continue
                values_by_key.setdefault(current_key, []).append(value)
                continue

            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip().strip('"').strip("'")
            if not key:
                current_key = None
                continue

            config[key] = value
            values_by_key.setdefault(key, []).append(value)
            current_key = key
        return config, values_by_key

    raw_lines = config_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        if stripped.startswith("---"):
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(raw_line)
    if current:
        blocks.append(current)

    global_config: dict[str, str] = {}
    jobs: List[dict[str, str]] = []

    job_marker_keys = {
        "reference",
        "ref",
        "source",
        "target",
        "video",
        "source_video",
        "sourcevideo",
        "source-video",
    }
    multi_job_keys = ["source", "target", "video", "source_video", "sourcevideo", "source-video"]

    for block in blocks:
        block_config, values_by_key = parse_block(block)
        if not block_config and not values_by_key:
            continue
        is_job = any(key in values_by_key for key in job_marker_keys)
        if is_job:
            expanded = False
            for multi_job_key in multi_job_keys:
                values = values_by_key.get(multi_job_key, [])
                if len(values) <= 1:
                    continue

                for value in values:
                    job_config = dict(block_config)
                    job_config[multi_job_key] = value
                    jobs.append(job_config)
                expanded = True
                break

            if not expanded:
                jobs.append(block_config)
        else:
            global_config.update(block_config)

    return global_config, jobs


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_det_size(value: str) -> tuple[int, int]:
    normalized = value.strip().lower().replace("x", " ").replace(",", " ")
    parts = [p for p in normalized.split() if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid det_size value (expected 2 ints): {value!r}")
    return int(parts[0]), int(parts[1])


def _parse_provider_list(value: str) -> list[str]:
    normalized = value.replace(";", ",").replace(" ", ",")
    providers = [p.strip() for p in normalized.split(",") if p.strip()]
    return providers


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _first_present(config: Mapping[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        lowered = key.lower()
        if lowered in config:
            return config[lowered]
    return None


def _apply_config_and_defaults(args: argparse.Namespace, config: Mapping[str, str], config_dir: Path) -> None:
    source_value = _first_present(config, "reference", "ref", "source_image", "sourceimage", "face", "image")
    target_value = _first_present(config, "source", "target", "video", "sourcevideo", "source_video", "source-video")
    output_value = _first_present(config, "output", "out")
    temp_output_value = _first_present(config, "temp_output", "temp-output", "tempoutput")
    providers_value = _first_present(config, "execution_providers", "execution-provider", "execution_provider", "providers", "provider")
    swap_all_value = _first_present(config, "swap_all_faces", "swap-all-faces", "swapallfaces")
    max_frames_value = _first_present(config, "max_frames", "max-frames", "maxframes")
    det_size_value = _first_present(config, "det_size", "det-size", "detsize")
    keep_temp_value = _first_present(config, "keep_temp", "keep-temp", "keeptemp")

    if getattr(args, "source", None) is None:
        args.source = _resolve_path(source_value, config_dir) if source_value else DEFAULT_SOURCE
    if getattr(args, "target", None) is None:
        args.target = _resolve_path(target_value, config_dir) if target_value else DEFAULT_TARGET

    if getattr(args, "output", None) is None:
        if output_value:
            args.output = _resolve_path(output_value, config_dir)
        else:
            suffix = args.target.suffix or ".mp4"
            args.output = args.target.with_name(f"{args.target.stem}-swapped{suffix}")

    if getattr(args, "temp_output", None) is None:
        if temp_output_value:
            args.temp_output = _resolve_path(temp_output_value, config_dir)
        else:
            suffix = args.target.suffix or ".mp4"
            args.temp_output = args.target.with_name(f".tmp-{args.target.stem}-swapped{suffix}")
    if getattr(args, "execution_providers", None) is None and providers_value:
        args.execution_providers = _parse_provider_list(providers_value)
    if getattr(args, "swap_all_faces", None) is None and swap_all_value is not None:
        args.swap_all_faces = _parse_bool(swap_all_value)
    if getattr(args, "max_frames", None) is None and max_frames_value is not None:
        args.max_frames = int(max_frames_value)
    if getattr(args, "det_size", None) is None and det_size_value is not None:
        args.det_size = _parse_det_size(det_size_value)
    if getattr(args, "keep_temp", None) is None and keep_temp_value is not None:
        args.keep_temp = _parse_bool(keep_temp_value)

    if args.max_frames is None:
        args.max_frames = 0
    if args.swap_all_faces is None:
        args.swap_all_faces = False
    if args.keep_temp is None:
        args.keep_temp = False
    if args.det_size is None:
        args.det_size = (640, 640)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config-swap-video.config"),
        help="Parameter file in Key=Value format (default: config-swap-video.config).",
    )
    pre_args, remaining = pre_parser.parse_known_args(argv)
    config_path = cast(Path, pre_args.config)
    global_config, job_configs = _load_kv_config_jobs(config_path)

    parser = argparse.ArgumentParser(description="Swap a face from an image into a video.", parents=[pre_parser])
    parser.add_argument("--source", type=Path, default=None, help="Source face image path.")
    parser.add_argument("--target", type=Path, default=None, help="Target video path.")
    parser.add_argument("--output", type=Path, default=None, help="Output video path.")
    parser.add_argument(
        "--temp-output",
        type=Path,
        default=None,
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
        default=None,
        help="Swap all detected faces in each frame instead of only the largest face.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames for testing. 0 means full video.",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Face detector input size.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=None,
        help="Keep the temporary video without remuxing cleanup.",
    )

    args = parser.parse_args(remaining)

    args._config_dir = config_path.parent
    args._config_global = global_config
    args._config_jobs = job_configs

    has_single_job_overrides = any(
        getattr(args, name, None) is not None for name in ("source", "target", "output", "temp_output")
    )
    if has_single_job_overrides or len(job_configs) <= 1:
        job_config = job_configs[0] if job_configs else {}
        effective_config = dict(global_config)
        effective_config.update(job_config)
        _apply_config_and_defaults(args, effective_config, config_path.parent)
        args.batch = False
    else:
        args.batch = True
    return args


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


def get_default_execution_providers() -> list[str]:
    available_providers = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


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

    providers = args.execution_providers or get_default_execution_providers()
    print(f"Using execution providers: {', '.join(providers)}")
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
    if not getattr(args, "batch", False):
        try:
            output_path = process_video(args)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(f"Output written to {output_path}")
        return 0

    config_dir = cast(Path, args._config_dir)
    global_config = cast(dict[str, str], args._config_global)
    job_configs = cast(List[dict[str, str]], args._config_jobs)

    failures = 0
    for index, job_config in enumerate(job_configs, start=1):
        effective_config = dict(global_config)
        effective_config.update(job_config)

        job_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if not k.startswith("_")})
        job_args.batch = False
        _apply_config_and_defaults(job_args, effective_config, config_dir)

        try:
            output_path = process_video(job_args)
        except Exception as exc:
            failures += 1
            print(f"Error (job {index}/{len(job_configs)}): {exc}", file=sys.stderr)
            continue

        print(f"Output written to {output_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())