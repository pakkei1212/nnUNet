#!/usr/bin/env python3
"""
End-to-end pipeline driver for the abdominal CT segmentation dataset.

This script performs dataset conversion to nnU-Net v2 format, experiment planning,
preprocessing, training, validation inference + evaluation, and test-time prediction.

Example:
    python main_pipeline.py \
        --data-root public_leaderboard_data \
        --dataset-id 500 \
        --dataset-name AbdominalCTMultiOrgan \
        --configurations 3d_fullres \
        --fold 0
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from skimage import io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data conversion, preprocessing, training, and inference for nnU-Net v2."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("public_leaderboard_data"),
        help="Root folder containing train_images, train_labels, val_images, val_labels, test1_images, spacing_mm.txt.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=500,
        help="Numeric ID (0-999) used to name the dataset folder (DatasetXXX_Name).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="AbdominalCTMultiOrgan",
        help="Dataset nickname appended to DatasetXXX_.",
    )
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=Path("./nnUNet_raw"),
        help="Destination for nnU-Net raw data (env: nnUNet_raw).",
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=Path("./nnUNet_preprocessed"),
        help="Destination for preprocessed data (env: nnUNet_preprocessed).",
    )
    parser.add_argument(
        "--nnunet-results",
        type=Path,
        default=Path("./nnUNet_results"),
        help="Destination for model checkpoints and inference outputs (env: nnUNet_results).",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        default=["3d_fullres"],
        help="nnU-Net configurations to preprocess/train (for example 3d_fullres, 2d).",
    )
    parser.add_argument(
        "--trainer-class",
        type=str,
        default="nnUNetTrainer",
        help="Trainer class name used for training.",
    )
    parser.add_argument(
        "--plans-identifier",
        type=str,
        default="nnUNetPlans",
        help="Planner identifier, controls the plan file name.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help="Fold index to train/evaluate (0-4 or 'all').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu", "mps"),
        help="Torch device for training/inference.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for training (>=1).",
    )
    parser.add_argument(
        "--num-processes-fingerprint",
        type=int,
        default=8,
        help="Processes for fingerprint extraction.",
    )
    parser.add_argument(
        "--num-processes-preprocess",
        type=int,
        default=8,
        help="Processes for preprocessing (per configuration).",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=None,
        help="Optional root folder to mirror predictions (defaults to nnUNet_results structure).",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="checkpoint_final.pth",
        help="Checkpoint file used for inference.",
    )
    parser.add_argument(
        "--planner-class",
        type=str,
        default="nnUNetPlannerResEncM",
        help="Experiment planner class to use (see documentation/resenc_presets.md for recommendations).",
    )
    parser.add_argument(
        "--preprocessor-class",
        type=str,
        default="DefaultPreprocessor",
        help="Preprocessor class name, see documentation for options.",
    )
    parser.add_argument(
        "--verify-dataset",
        action="store_true",
        help="Verify dataset integrity during fingerprint extraction.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip raw dataset conversion if data already exists.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip planning & preprocessing.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training.",
    )
    parser.add_argument(
        "--skip-validation-inference",
        action="store_true",
        help="Skip validation inference and evaluation.",
    )
    parser.add_argument(
        "--skip-test-inference",
        action="store_true",
        help="Skip test set inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files when rerunning stages.",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Also export softmax probabilities during inference.",
    )
    parser.add_argument(
        "--export-validation-probabilities",
        action="store_true",
        help="Save validation softmax outputs during training (enables nnUNetv2_find_best_configuration).",
    )
    parser.add_argument(
        "--bounding-box-prompts",
        type=Path,
        default=None,
        help="Optional path to test bounding box prompts (stored alongside predictions for reference).",
    )
    parser.add_argument(
        "--only-configuration",
        type=str,
        default=None,
        help="Restrict training/inference to a single configuration from --configurations.",
    )
    parser.add_argument(
        "--no-log",
        dest="log_to_stdout",
        action="store_false",
        help="Disable logging messages printed by the pipeline stages.",
    )
    parser.set_defaults(log_to_stdout=True)
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("nnUNet_raw", str(args.nnunet_raw.resolve()))
    os.environ.setdefault("nnUNet_preprocessed", str(args.nnunet_preprocessed.resolve()))
    os.environ.setdefault("nnUNet_results", str(args.nnunet_results.resolve()))


def ensure_dependencies() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "PyTorch is required to run the pipeline. Please install the project dependencies first."
        ) from exc


def parse_spacing_map(spacing_file: Path) -> Dict[str, Tuple[float, float, float]]:
    if not spacing_file.exists():
        raise FileNotFoundError(f"Spacing file not found: {spacing_file}")
    mapping: Dict[str, Tuple[float, float, float]] = {}
    with spacing_file.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            case_id = key.strip().zfill(2)
            spacing = ast.literal_eval(value.strip())
            if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
                raise ValueError(f"Unexpected spacing entry for case {case_id}: {value}")
            mapping[case_id] = tuple(float(v) for v in spacing)
    return mapping


def sorted_slice_paths(case_folder: Path) -> List[Path]:
    slices = sorted(case_folder.glob("*.png"), key=lambda p: int(p.stem))
    if not slices:
        raise FileNotFoundError(f"No PNG slices found in {case_folder}")
    return slices


def load_stack(slice_paths: Sequence[Path]) -> np.ndarray:
    stack = [io.imread(str(p)) for p in slice_paths]
    volume = np.stack(stack, axis=0)
    return volume


def write_nifti(volume: np.ndarray, spacing: Tuple[float, float, float], output_path: Path, dtype: np.dtype) -> None:
    img = sitk.GetImageFromArray(volume.astype(dtype, copy=False))
    img.SetSpacing(tuple(float(v) for v in spacing))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))


def convert_split_to_nnunet(
    case_ids: Iterable[str],
    image_root: Path,
    label_root: Optional[Path],
    output_images: Path,
    output_labels: Optional[Path],
    spacing_map: Dict[str, Tuple[float, float, float]],
    prefix: str,
    overwrite: bool,
) -> List[str]:
    case_identifiers: List[str] = []
    for case_id in sorted(case_ids, key=lambda x: int(x)):
        image_case_dir = image_root / case_id
        label_case_dir = label_root / case_id if label_root is not None else None
        if not image_case_dir.is_dir():
            raise FileNotFoundError(f"Missing image folder for case {case_id}: {image_case_dir}")
        spacing = spacing_map.get(case_id)
        if spacing is None:
            raise KeyError(f"No spacing metadata for case {case_id} in spacing_mm.txt")
        case_name = f"{prefix}_{case_id.zfill(3)}"
        image_output_path = output_images / f"{case_name}_0000.nii.gz"
        if image_output_path.exists() and not overwrite:
            case_identifiers.append(case_name)
            continue
        slices = sorted_slice_paths(image_case_dir)
        volume = load_stack(slices).astype(np.int16, copy=False)
        write_nifti(volume, spacing, image_output_path, np.int16)
        if label_case_dir is not None:
            if output_labels is None:
                raise ValueError("Label root provided but output label directory missing.")
            label_slices = sorted_slice_paths(label_case_dir)
            if len(label_slices) != len(slices):
                raise ValueError(
                    f"Mismatched slice count for case {case_id}: {len(slices)} images vs {len(label_slices)} labels"
                )
            label_volume = load_stack(label_slices).astype(np.uint8, copy=False)
            label_output_path = output_labels / f"{case_name}.nii.gz"
            write_nifti(label_volume, spacing, label_output_path, np.uint8)
        case_identifiers.append(case_name)
    return case_identifiers


def parse_bbox_prompts(bbox_file: Path, output_json: Path) -> None:
    if not bbox_file or not bbox_file.exists():
        return
    prompts = {}
    with bbox_file.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("<") or ">:" not in line:
                continue
            key, value = line.split(">:")
            triplet = key.strip("<>").split(",")
            if len(triplet) != 3:
                continue
            case_id = triplet[0].strip().zfill(2)
            slice_idx = triplet[1].strip()
            organ_idx = triplet[2].strip()
            coords = ast.literal_eval(value.strip())
            prompts.setdefault(case_id, {}).setdefault(slice_idx, {})[organ_idx] = coords
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(prompts, f, indent=2)


def generate_dataset_json_file(
    dataset_dir: Path,
    num_training_cases: int,
    channel_label: str,
    labels: Dict[str, int],
    file_ending: str,
    dataset_name: str,
    metadata: Dict[str, object],
) -> None:
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

    generate_dataset_json(
        str(dataset_dir),
        channel_names={0: channel_label},
        labels=labels,
        num_training_cases=num_training_cases,
        file_ending=file_ending,
        dataset_name=dataset_name,
        **metadata,
    )


def prepare_raw_dataset(args: argparse.Namespace, dataset_dir: Path) -> Dict[str, List[str]]:
    spacing_map = parse_spacing_map(args.data_root / "spacing_mm.txt")
    train_ids = [p.name for p in (args.data_root / "train_images").iterdir() if p.is_dir()]
    val_ids = [p.name for p in (args.data_root / "val_images").iterdir() if p.is_dir()]
    test_ids = [p.name for p in (args.data_root / "test1_images").iterdir() if p.is_dir()]

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    train_cases = convert_split_to_nnunet(
        train_ids,
        args.data_root / "train_images",
        args.data_root / "train_labels",
        images_tr,
        labels_tr,
        spacing_map,
        prefix="ct",
        overwrite=args.overwrite,
    )
    val_cases = convert_split_to_nnunet(
        val_ids,
        args.data_root / "val_images",
        args.data_root / "val_labels",
        images_tr,
        labels_tr,
        spacing_map,
        prefix="ct",
        overwrite=args.overwrite,
    )
    test_cases = convert_split_to_nnunet(
        test_ids,
        args.data_root / "test1_images",
        None,
        images_ts,
        None,
        spacing_map,
        prefix="ct",
        overwrite=args.overwrite,
    )
    metadata = {
        "training_cases": train_cases,
        "validation_cases": val_cases,
        "test_cases": test_cases,
        "spacing_file": str((args.data_root / "spacing_mm.txt").resolve()),
    }
    labels = {"background": 0}
    for organ_idx in range(1, 13):
        labels[f"organ_{organ_idx:02d}"] = organ_idx
    generate_dataset_json_file(
        dataset_dir=dataset_dir,
        num_training_cases=len(train_cases) + len(val_cases),
        channel_label="CT",
        labels=labels,
        file_ending=".nii.gz",
        dataset_name=dataset_dir.name,
        metadata=metadata,
    )

    splits = [{"train": train_cases, "val": val_cases}]
    splits_file = dataset_dir / "splits_final.json"
    if not splits_file.exists() or args.overwrite:
        with splits_file.open("w") as f:
            json.dump(splits, f, indent=2)

    if args.bounding_box_prompts:
        parse_bbox_prompts(args.bounding_box_prompts, dataset_dir / "test_bboxes.json")

    return {"train": train_cases, "val": val_cases, "test": test_cases}


def run_planning_and_preprocessing(
    dataset_id: int,
    plans_identifier: str,
    configurations: Sequence[str],
    num_proc_fp: int,
    num_proc_preprocess: int,
    planner_class: str,
    preprocessor_class: str,
    verify_dataset: bool,
) -> str:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints,
        plan_experiments,
        preprocess,
    )

    dataset_ids = [dataset_id]
    extract_fingerprints(
        dataset_ids,
        num_processes=num_proc_fp,
        check_dataset_integrity=verify_dataset,
        clean=True,
        verbose=True,
    )
    resulting_plans_identifier = plan_experiments(
        dataset_ids,
        experiment_planner_class_name=planner_class,
        preprocess_class_name=preprocessor_class,
    )
    preprocess(
        dataset_ids,
        plans_identifier=resulting_plans_identifier or plans_identifier,
        configurations=tuple(configurations),
        num_processes=tuple([num_proc_preprocess] * len(configurations)),
        verbose=True,
    )
    return resulting_plans_identifier or plans_identifier


def run_training_stage(
    dataset_name: str,
    configuration: str,
    trainer_class: str,
    plans_identifier: str,
    fold: str,
    num_gpus: int,
    device: str,
    export_validation_probabilities: bool,
) -> None:
    import torch
    from nnunetv2.run.run_training import run_training

    torch_device = torch.device(device)
    run_training(
        dataset_name,
        configuration=configuration,
        fold=fold,
        trainer_class_name=trainer_class,
        plans_identifier=plans_identifier,
        num_gpus=num_gpus,
        device=torch_device,
        export_validation_probabilities=export_validation_probabilities,
    )


def build_model_output_dir(dataset_name: str, trainer_class: str, plans_identifier: str, configuration: str) -> Path:
    base = Path(os.environ["nnUNet_results"])
    return base / dataset_name / f"{trainer_class}__{plans_identifier}__{configuration}"


def run_inference(
    model_dir: Path,
    fold: str,
    inputs: Sequence[List[str]],
    output_dir: Path,
    device: str,
    checkpoint_name: str,
    save_probabilities: bool,
    overwrite: bool,
) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(device=torch.device(device))
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files(
        list(inputs),
        str(output_dir),
        save_probabilities=save_probabilities,
        overwrite=overwrite,
    )


def compute_validation_metrics(
    predictions_dir: Path,
    dataset_dir: Path,
    plans_identifier: str,
    output_filename: Optional[Path] = None,
) -> Dict[str, object]:
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2

    dataset_json = dataset_dir / "dataset.json"
    plans_file = Path(os.environ["nnUNet_preprocessed"]) / dataset_dir.name / f"{plans_identifier}.json"
    gt_folder = dataset_dir / "labelsTr"
    summary = compute_metrics_on_folder2(
        str(gt_folder),
        str(predictions_dir),
        str(dataset_json),
        str(plans_file),
        output_file=str(output_filename) if output_filename else None,
    )
    return summary


def build_inference_input_lists(image_dir: Path, case_ids: Sequence[str], file_ending: str) -> List[List[str]]:
    inputs: List[List[str]] = []
    for case in case_ids:
        image_file = image_dir / f"{case}_0000{file_ending}"
        if not image_file.exists():
            raise FileNotFoundError(f"Missing input volume for inference: {image_file}")
        inputs.append([str(image_file)])
    return inputs


def main() -> None:
    args = parse_args()
    configure_environment(args)
    ensure_dependencies()

    if args.log_to_stdout:
        print("== nnU-Net pipeline starting ==")

    for root in (args.nnunet_raw, args.nnunet_preprocessed, args.nnunet_results):
        root.mkdir(parents=True, exist_ok=True)

    dataset_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = Path(os.environ["nnUNet_raw"]) / dataset_folder_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    case_splits: Dict[str, List[str]]
    if args.skip_conversion and (dataset_dir / "dataset.json").exists():
        with (dataset_dir / "dataset.json").open("r") as f:
            dataset_meta = json.load(f)
        case_splits = {
            "train": dataset_meta.get("training_cases", []),
            "val": dataset_meta.get("validation_cases", []),
            "test": dataset_meta.get("test_cases", []),
        }
        if args.log_to_stdout:
            print("Skipping dataset conversion (dataset.json already present).")
    else:
        if args.log_to_stdout:
            print("Converting raw dataset into nnU-Net format...")
        case_splits = prepare_raw_dataset(args, dataset_dir)
        if args.log_to_stdout:
            print(f"Converted dataset stored at {dataset_dir}")

    active_configurations = args.configurations
    if args.only_configuration is not None:
        active_configurations = [args.only_configuration]

    if not args.skip_preprocessing:
        if args.log_to_stdout:
            print("Running nnU-Net experiment planning & preprocessing...")
        resolved_plans_identifier = run_planning_and_preprocessing(
            dataset_id=args.dataset_id,
            plans_identifier=args.plans_identifier,
            configurations=active_configurations,
            num_proc_fp=args.num_processes_fingerprint,
            num_proc_preprocess=args.num_processes_preprocess,
            planner_class=args.planner_class,
            preprocessor_class=args.preprocessor_class,
            verify_dataset=args.verify_dataset,
        )
        plans_identifier = resolved_plans_identifier
    else:
        plans_identifier = args.plans_identifier
        if args.log_to_stdout:
            print("Skipping planning & preprocessing.")

    dataset_name = dataset_folder_name
    file_ending = ".nii.gz"

    for configuration in active_configurations:
        model_dir = build_model_output_dir(dataset_name, args.trainer_class, plans_identifier, configuration)
        if not args.skip_training:
            if args.log_to_stdout:
                print(f"Starting training for configuration {configuration} (fold {args.fold})...")
            run_training_stage(
                dataset_name=dataset_name,
                configuration=configuration,
                trainer_class=args.trainer_class,
                plans_identifier=plans_identifier,
                fold=args.fold,
                num_gpus=args.num_gpus,
                device=args.device,
                export_validation_probabilities=args.export_validation_probabilities,
            )
        else:
            if args.log_to_stdout:
                print(f"Skipping training for configuration {configuration}.")

        fold_dir = model_dir / f"fold_{args.fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(
                f"Expected trained fold directory does not exist: {fold_dir}. Did training finish successfully?"
            )

        if not args.skip_validation_inference and case_splits.get("val"):
            if args.log_to_stdout:
                print("Running validation inference...")
            val_inputs = build_inference_input_lists(
                dataset_dir / "imagesTr",
                case_splits["val"],
                file_ending=file_ending,
            )
            val_output_dir = (
                args.prediction_output / configuration / "val"
                if args.prediction_output
                else fold_dir / "pipeline_val_predictions"
            )
            run_inference(
                model_dir=model_dir,
                fold=args.fold,
                inputs=val_inputs,
                output_dir=val_output_dir,
                device=args.device,
                checkpoint_name=args.checkpoint_name,
                save_probabilities=args.save_probabilities,
                overwrite=args.overwrite,
            )
            summary_file = val_output_dir / "summary.json"
            summary = compute_validation_metrics(
                predictions_dir=val_output_dir,
                dataset_dir=dataset_dir,
                plans_identifier=plans_identifier,
                output_filename=summary_file,
            )
            if args.log_to_stdout:
                dice_scores = summary["foreground_mean"]["Dice"]
                print(f"Validation foreground Dice (mean over organs): {dice_scores:.4f}")
                print(f"Validation metrics saved to {summary_file}")
        elif args.log_to_stdout:
            print("Skipping validation inference (no validation cases or flag set).")

        if not args.skip_test_inference and case_splits.get("test"):
            if args.log_to_stdout:
                print("Running test inference...")
            test_inputs = build_inference_input_lists(
                dataset_dir / "imagesTs",
                case_splits["test"],
                file_ending=file_ending,
            )
            test_output_dir = (
                args.prediction_output / configuration / "test"
                if args.prediction_output
                else fold_dir / "pipeline_test_predictions"
            )
            run_inference(
                model_dir=model_dir,
                fold=args.fold,
                inputs=test_inputs,
                output_dir=test_output_dir,
                device=args.device,
                checkpoint_name=args.checkpoint_name,
                save_probabilities=args.save_probabilities,
                overwrite=args.overwrite,
            )
            if args.bounding_box_prompts:
                bbox_target = test_output_dir / "test_bboxes.json"
                if not bbox_target.exists():
                    parse_bbox_prompts(args.bounding_box_prompts, bbox_target)
            if args.log_to_stdout:
                print(f"Test predictions saved to {test_output_dir}")
        elif args.log_to_stdout:
            print("Skipping test inference.")

    if args.log_to_stdout:
        print("== nnU-Net pipeline finished ==")


if __name__ == "__main__":
    main()
