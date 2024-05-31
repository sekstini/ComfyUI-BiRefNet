import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.profiler import profile, ProfilerActivity
from torch.utils.benchmark import Timer

from config import models_dir
from models.baseline import BiRefNet
from image_processor import BiRefNetImageProcessor


# Ensure reproducibility
torch.manual_seed(0)

def load_model(ckpt: str = "BiRefNet-DIS_ep580.pth"):
    model = BiRefNet().eval()
    state_dict = torch.load(os.path.join(models_dir, ckpt))

    # Fix keys since we no longer set norms directly on the module
    state_dict = {k.replace("bb.norm", "bb.norms."): v for k, v in state_dict.items()}

    # Model produces garbage outputs unless we filter out these keys
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to("cuda", torch.float16)
    return model


model = load_model()
image_processor = BiRefNetImageProcessor(model.config.size)

@torch.inference_mode()
def infer(pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", torch.float16, enabled=False):
        return model(pixel_values)


img = Image.open("green_knight.jpeg")
pixel_values = image_processor(img).to("cuda", torch.float16).unsqueeze(0)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    # profile_memory=True,
    # record_shapes=True,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
) as prof:
    for _ in range(5):
        outputs = infer(pixel_values)
        torch.cuda.synchronize()
        prof.step()


git_commit = os.popen("git rev-parse HEAD").read().strip()
benchmark_dir = Path(f"benchmarks/{git_commit}")
benchmark_dir.mkdir(parents=True, exist_ok=True)
prof.export_chrome_trace(str(benchmark_dir / "trace.json"))
# prof.export_memory_timeline(str(benchmark_dir / "memory.html"), "cuda:0")


print("Benchmarking inference...")
measurement = Timer(
    stmt="infer(pixel_values)",
    globals={"infer": infer, "pixel_values": pixel_values},
    label="BiRefNet",
).blocked_autorange(min_run_time=2)
print(measurement, file=(benchmark_dir / "benchmark.txt").open("w"))


outputs = infer(pixel_values)
outputs = F.interpolate(outputs[-1].sigmoid(), size=img.size, mode="bilinear", align_corners=True)
out = outputs[0].cpu().round().mul(255.0).to(torch.uint8).squeeze()
segmentation_img = Image.fromarray(out.numpy()).convert("RGB")
segmentation_img.save(benchmark_dir / "processed_image.png")
