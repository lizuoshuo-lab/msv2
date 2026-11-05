# Third-party sources

This repository provides the deployment and video-to-scene workflow. The installer downloads the following pinned upstream sources and retains their original license files under the runtime `vendor` directory. Videos, pretrained weights, generated scenes and compiled CUDA libraries are not included in this Git repository.

| Component | Source | License and conditions |
| --- | --- | --- |
| Mini-Splatting2 | [fatPeter/mini-splatting2](https://github.com/fatPeter/mini-splatting2), commit `5ec9202c2db7d900728bfc09733e85467cff885e` | Upstream root `LICENSE`: Apache 2.0. Its bundled rasterization and nearest-neighbor components carry separate notices and conditions; the root license does not replace them. |
| Gaussian rasterizers and simple-knn | Mini-Splatting2 `submodules` | Rasterizer headers and `LICENSE.md` retain Inria / GRAPHDECO's non-commercial research and evaluation conditions. Review the upstream licenses before commercial use. A CUDA 12.8 compatibility change adds only the standard `<cfloat>` header in the isolated runtime copy of `simple_knn.cu`; both hashes are recorded. |
| fused-ssim | Mini-Splatting2 `submodules/fused-ssim` | MIT, Copyright 2024 Rahul Goel; upstream license retained. |
| GLM | Rasterizer `third_party/glm` | Upstream GLM license files are retained unchanged. |
| SwinIR | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR), commit `6545850fbf8df298df73d81f3e8cba638787c8bd` | Apache 2.0; review upstream notices and pretrained-model conditions. The official real-image 4x large model is downloaded separately and verified by SHA256. |
| PyTorch / torchvision | [pytorch.org](https://pytorch.org/) | Their original licenses and bundled third-party notices apply. |
| COLMAP / FFmpeg | Distribution packages | Their respective licenses and build-dependent component conditions apply. |

Source commits, archive hashes, model download URL and CUDA/PyTorch versions are recorded in `upstream.lock.json`. Deployment records the actual runtime installation separately. Training uses the upstream sparse MSV2 optimizer and loss; these components are not relicensed by this workflow.
