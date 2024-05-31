import os
from pathlib import Path

from numpy import dtype
import torch
import torch_tensorrt
import torch.nn.functional as F
from PIL import Image
from torch.profiler import profile, ProfilerActivity
from torch.utils.benchmark import Timer
import torch_tensorrt._enums

from config import models_dir
from models.baseline import BiRefNet
from image_processor import BiRefNetImageProcessor

import torch._dynamo.config
import torch._inductor.config

# Ensure reproducibility
torch.manual_seed(0)

def load_model(ckpt: str = "BiRefNet-DIS_ep580.pth"):
    model = BiRefNet()
    state_dict = torch.load(os.path.join(models_dir, ckpt))

    # Fix keys since we no longer set norms directly on the module
    state_dict = {k.replace("bb.norm", "bb.norms."): v for k, v in state_dict.items()}

    # Model produces garbage outputs unless we filter out these keys
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model.to("cuda", torch.float16).eval().requires_grad_(False)

model = load_model()
image_processor = BiRefNetImageProcessor(model.config.size)

img = Image.open("green_knight.jpeg")
pixel_values = image_processor(img).to("cuda", torch.float16).unsqueeze(0)

"""
optimized_model = torch_tensorrt.compile(
    model,
    ir="torch_compile",
    #inputs=[torch_tensorrt.Input(
    #    min_shape=(1, 3, 512, 512),
    #    opt_shape=(1, 3, 1024, 1024),
    #    max_shape=(1, 3, 1024, 1024),
    #    dtype=torch.float16,
    #    tensor_domain=(-1.0, 1.0),
    #)],
    inputs=[pixel_values],
    enabled_precisions={torch.float16},
    debug=True,
    workspace_size=20 << 30,
    min_block_size=1024,
    optimization_level=4,
    enable_experimental_decompositions=False,
)
print("Optimized model compiled")
"""

@torch.no_grad()
def infer(pixel_values: torch.Tensor) -> torch.Tensor:
    return model(pixel_values)

git_commit = os.popen("git rev-parse HEAD").read().strip()
benchmark_dir = Path(f"benchmarks/{git_commit}")
benchmark_dir.mkdir(parents=True, exist_ok=True)

if True:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        # profile_memory=True,
        # record_shapes=True,
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3),
    ) as prof:
        for _ in range(5):
            outputs = infer(pixel_values)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(str(benchmark_dir / "trace.json"))
    # prof.export_memory_timeline(str(benchmark_dir / "memory.html"), "cuda:0")


# Warmup outside of profiler to avoid recompiles in benchmark
infer(pixel_values)

print("Benchmarking inference...")
measurement = Timer(
    stmt="infer(pixel_values)",
    globals={"infer": infer, "pixel_values": pixel_values},
    label="BiRefNet",
).blocked_autorange(min_run_time=2)
print(measurement, file=(benchmark_dir / "benchmark.txt").open("w"))


outputs = infer(pixel_values)
print(outputs[-1])
outputs = F.interpolate(outputs[-1].sigmoid(), size=img.size, mode="bilinear", align_corners=True)
out = outputs[0].cpu().round().mul(255.0).to(torch.uint8).squeeze()
segmentation_img = Image.fromarray(out.numpy()).convert("RGB")
segmentation_img.save(benchmark_dir / "processed_image.png")
