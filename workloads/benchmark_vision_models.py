import argparse

from collections import namedtuple, Counter, defaultdict
from collections.abc import Callable

import json
import os

import statistics
from typing import Dict, List, Set
import timeit

from .common import write_results

from src import mlir_parser
from src.stablehlo_op_wrapper import StableHLOOpWrapper

import numpy as np

from mlir import ir
from mlir import passmanager as pm
from mlir.dialects import stablehlo

import jax

import torch
from torch.export import export

import torchax
import torchax.export

from datasets import load_dataset

from transformers import AutoImageProcessor
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    ConvNextV2ForImageClassification,
    CLIPForImageClassification,
    LevitForImageClassification,
    MobileViTV2Config,
    ResNetForImageClassification,
    SegformerForSemanticSegmentation,
    Swinv2ForImageClassification,
    ViTForImageClassification,
)

import shutil

PyTorchModel = namedtuple("PyTorchModel", "model inputs")


def clip(image) -> PyTorchModel:
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def convnextv2(image) -> PyTorchModel:
    model_id = "facebook/convnextv2-tiny-1k-224"
    model = ConvNextV2ForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def levit(image) -> PyTorchModel:
    model_id = "facebook/levit-128S"
    model = LevitForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def mobilevit2(image) -> PyTorchModel:
    # Using the timm version for mobilevit2 because the model from Apple requires a token
    model = AutoModelForImageClassification.from_config(MobileViTV2Config())
    image_processor = AutoImageProcessor.from_pretrained(
        "timm/mobilevitv2_100.cvnets_in1k"
    )

    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def resnet50(image) -> PyTorchModel:
    model_id = "microsoft/resnet-50"
    model = ResNetForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def segformer(image) -> PyTorchModel:
    model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
    model = SegformerForSemanticSegmentation(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(images=image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def swinv2(image) -> PyTorchModel:
    model_id = "microsoft/swinv2-tiny-patch4-window8-256"
    model = Swinv2ForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(images=image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def vitbase(image) -> PyTorchModel:
    model_id = "google/vit-base-patch16-224"
    model = ViTForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(images=image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def get_unique_stable_hlo_ops(all_ops: List[StableHLOOpWrapper]) -> Counter:
    res = Counter()

    for wrapped_op in all_ops:
        res[wrapped_op.op.name] += 1
    return res


def jsonify_results(num_runs: int, results: Dict[str, Dict[str, object]]) -> str:
    json_obj = {"numRuns": num_runs}
    models = []

    grouped_by_model = defaultdict(list)
    for _, obj in results.items():
        grouped_by_model[obj["model"]].append(obj)

    for model, res in grouped_by_model.items():
        model_obj = {"model": model, "results": res}
        models.append(model_obj)

    json_obj["models"] = models
    return json.dumps(json_obj, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Script for running inference on hugging face vision models"
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=[],
        help="Specify the models to run inference on. Defaults to running all models if this field is not set.",
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Specify how many inference runs to measure for a model",
    )

    parser.add_argument("-d", "--device", type=str, required=True)

    parser.add_argument("--export-stable-hlo", action=argparse.BooleanOptionalAction)

    parser.add_argument("--profile-jax", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    models: Set[str] = set(args.models)
    num_warmup_runs = 2

    models_map: Dict[str, Callable[[str], PyTorchModel]] = {
        "clip": clip,
        "convnextv2": convnextv2,
        "levit": levit,
        "mobilevit2": mobilevit2,
        "resnet50": resnet50,
        "segformer": segformer,
        "swinv2": swinv2,  # TODO: can't successfully export to stablehlo
        "vitbase": vitbase,
    }

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    if not models:
        models = models_map.keys()

    total_unique_stable_hlo_ops = Counter()
    unique_ops_per_model: Dict[str, Counter] = {}

    pipeline = ["stablehlo-refine-shapes,canonicalize,cse"]
    results: Dict[str, Dict[str, object]] = defaultdict(dict)

    batch_sizes = [1, 8, 16, 64, 128, 256, 512, 1024]
    timeit_repeat = 3

    for model in models:
        torch_model_and_inputs: PyTorchModel = models_map[model](image)

        torch_model = torch_model_and_inputs.model.eval()

        if args.export_stable_hlo and model != "swinv2":
            for batch_size in batch_sizes:
                print(
                    f"Exporting {model} with size {batch_size} to stablehlo and optimized HLO"
                )
                torch_inputs = torch_model_and_inputs.inputs["pixel_values"].repeat(
                    batch_size, 1, 1, 1
                )
                exported = export(torch_model, (torch_inputs,))

                # Write optimized HLO
                weights, func = torchax.export.exported_program_to_jax(exported)
                jax_inputs = jax.numpy.array(torch_inputs.numpy())
                hlo = jax.jit(func).lower(weights, jax_inputs).compile().as_text()
                with open(f"models/{model}-{'-'.join(map(str, list(torch_inputs.shape)))}.hlo", "w") as f:
                    f.write(hlo)

                # Write StableHLO
                _, stablehlo_obj = torchax.export.exported_program_to_stablehlo(
                    exported
                )
                mlir_module_string = stablehlo_obj.mlir_module()

                with ir.Context() as context:
                    stablehlo.register_dialect(context)

                    mlir_module = ir.Module.parse(mlir_module_string, context=context)
                    stablehlo.register_stablehlo_passes()

                    pass_manager = pm.PassManager.parse(
                        f"builtin.module({','.join(pipeline)})"
                    )
                    pass_manager.run(mlir_module.operation)

                    parsed_ops = mlir_parser.parse(mlir_module)
                    ops_counter = get_unique_stable_hlo_ops(parsed_ops)
                    unique_ops_per_model[model] = ops_counter
                    total_unique_stable_hlo_ops.update(ops_counter)

                    with open(f"models/{model}-{'-'.join(map(str, list(torch_inputs.shape)))}.mlir", "w") as f:
                        f.write(stablehlo_obj.mlir_module())

        torchax.enable_globally()
        torch_model = torch_model.to("jax")

        def forward_wrapper(inputs):
            return torch_model(inputs).logits

        model_jitted = torchax.compile(forward_wrapper)

        for batch_size in batch_sizes:
            if device == "tpuv5e" and (
                (model == "swinv2" and batch_size == 1024)
                or (model == "segformer" and batch_size > 128)
            ):
                continue

            torch_inputs = (
                torch_model_and_inputs.inputs["pixel_values"]
                .repeat(batch_size, 1, 1, 1)
                .to("jax")
            )
            print(
                f"Running inference on {model} with input shape: {torch_inputs.shape}"
            )

            try:
                print(f"Warming up {model} with {num_warmup_runs} runs")
                for i in range(num_warmup_runs):
                    res = model_jitted(torch_inputs)
                    res.apply_jax_(jax.block_until_ready)

                latencies_ms = []
                print(f"Performing {args.num_iterations} runs for {model}")
                key = f"{model}-{'-'.join(map(str, list(torch_inputs.shape)))}"

                profile_path = f"profiles/{args.device}/{key}"

                if os.path.isdir(profile_path):
                    shutil.rmtree(profile_path)  # start afresh

                def run_benchmark():
                    res = model_jitted(torch_inputs)
                    res.apply_jax_(jax.block_until_ready)

                with jax.profiler.trace(profile_path):
                    latencies_s = timeit.repeat(
                        run_benchmark, repeat=timeit_repeat, number=args.num_iterations
                    )

                latencies_ms = [(s * 1000) / args.num_iterations for s in latencies_s]
                results[key] = {
                    "model": model,
                    "key": key,
                    "inputShape": str(list(torch_inputs.shape)),
                    "mean": statistics.mean(latencies_ms),
                    "median": statistics.median(latencies_ms),
                    "standardDeviation": statistics.stdev(latencies_ms),
                    "min": min(latencies_ms),
                }

                print(
                    f"Results for {model} with input shape {torch_inputs.shape}: {latencies_ms}"
                )
            except (jax.errors.JaxRuntimeError, jax.errors.UnexpectedTracerError) as e:
                print(
                    f"Skipping shape {torch_inputs.shape} for {model} due to error: {e}"
                )

        torchax.disable_globally()

    json_results = jsonify_results(args.num_iterations * timeit_repeat, results)
    write_results(json_results, f"benchmarks/models/{args.device}")

    if args.export_stable_hlo:
        print(f"Unique ops per model: {unique_ops_per_model}")
        print(f"Overall unique ops: {total_unique_stable_hlo_ops}")
