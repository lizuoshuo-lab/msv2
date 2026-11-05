# msv2

将视频或标准 COLMAP 数据集重建为通用 3D Gaussian Splatting 场景。适用于静态场景、近似静止人物与移动相机的视频，不依赖 SMPL-X 或专用查看器。

## 部署与训练

在 Ubuntu 22.04、Python 3.10/3.11、NVIDIA 驱动和 CUDA Toolkit 12.8 的机器上运行：

```bash
git clone https://github.com/lizuoshuo-lab/msv2.git ~/msv2/app && bash ~/msv2/app/deploy.sh
```

出现 `MSV2_DEPLOYED` 后，将**完整上传的视频**放进 `~/msv2/videos/`，自动开始训练。支持 MP4、MOV、MKV、AVI、M4V、WebM。WinSCP 可使用临时文件名上传，完成后自动改名；未完成的 `.filepart` 文件不会进入训练。其他上传工具建议先用 `.part` 后缀，完成后再改为视频后缀。

源码、CUDA 扩展、公开 SwinIR 权重由部署命令在目标机器上获取、校验并安装；视频、权重和结果都在 Git 之外。缺失的系统工具在有 root 或免密 sudo 权限时自动安装，否则命令会给出需要安装的包。部署不安装显卡驱动，不替换其他项目的 Python 环境。

查看状态：

```bash
bash ~/msv2/app/msv2.sh status
```

也可手动训练任意路径的视频，或已有的 COLMAP 数据集：

```bash
bash ~/msv2/app/msv2.sh train /path/to/video.mp4
bash ~/msv2/app/msv2.sh train /path/to/colmap-dataset --scales 1,2,4
```

COLMAP 目录应包含 `images/` 和 `sparse/0/`，支持文本或二进制模型，以及已去畸变的 `PINHOLE`、`SIMPLE_PINHOLE` 相机。包含畸变参数的模型需先通过 COLMAP `image_undistorter` 转为标准输入。导入时保留相机、稀疏点和观测关系，统一图像名，避免多相机文件重名。

## 默认流程

1. 视频按真实 PTS 以 3 fps 选择最近源帧，去除重复选择；保留完整画面和显示方向，支持横屏、竖屏及旋转元数据。
2. COLMAP 从图像恢复相机和稀疏点；必须全部采样帧注册成功。不用人体姿态推测相机，不插值补位姿。
3. 从原始完整 RGB 分别生成原始尺寸、Lanczos 插值两倍、SwinIR 超分四倍输入；背景一起放大和训练，无 mask、无人物裁剪。
4. 相机内参按实际倍率完整缩放，外参不变。每个倍率重新初始化高斯及优化器。
5. 使用固定上游 MSV2 的 `config/fast`，`outdoor`，种子 0，18,000 步。保留原生 3,000 步半尺寸预热、2,000 步深度重初始化和 3,000／8,000 步简化；不额外限制点数，不在训练后删点。
6. 核验原生 PLY 回读、坐标变换后的协方差／SH／渲染一致性，导出通用文件，并在原图分辨率下比较同一组视角。

预热发生在原始／放大输入准备完成后；没有“先将原图缩小，再做两倍或四倍放大”的输入步骤。帧数、分辨率和相机参数均来自各自数据集，不固定为某个视频或 59 帧。

配置在 `~/msv2/config.json`，首次部署从 [默认配置](configs/default.json) 创建，更新时保留。可修改 `sample_fps`、`scales`、`importance_metric` 等；`frame_indices` 可指定递增、唯一的原视频帧号以精确复现某个数据集。训练预算需至少 8,001 步，保证完整走过原生简化阶段；默认是 18,000。完整选项见 [配置定义](msv2/config.py)。修改配置后下一项任务会采用新配置，已有任务保持冻结配置。

## 结果和日志

每次训练位于 `~/msv2/runs/<名称>/`：

- `state.json`：本轮阶段、错误和已完成倍率。
- `logs/`：抽帧、相机、预检、训练和导出的逐阶段日志；预处理详细日志另在 `prepared/logs/`。
- `export/comparison.html`：原图与各倍率逐帧对照，留出指标统一在原图尺寸计算。
- `export/<倍率>/scene.zip`：标准高斯 `point_cloud.ply`、完整 `cameras.json`、COLMAP 文本相机、坐标变换与一致性核验记录。
- `export/<倍率>/point_cloud_world.ply`、`cameras_world.json`：原训练世界坐标文件。

主 PLY 将 COLMAP 世界坐标绕 X 轴旋转 180°，中心、完整协方差、各阶 SH 和相机同步变换；它不代表测量得到的重力方向。文件采用标准 3DGS 字段，可独立导入支持该格式的查看器。COLMAP 文本导出提供相机轨迹，不包含原始特征观测。

同一时间只运行一项任务。自动队列按视频内容、配置和实现版本去重；手动 `train` 每次创建全新运行。较大倍率显存不足时会保留已完成倍率，状态明确为 `partial`，不会偷偷缩小图像或把失败记成成功。修复输入或环境后可用手动命令重训。格式和训练检查通过不等于新视角几何质量通过。

更新：

```bash
git -C ~/msv2/app pull --ff-only && bash ~/msv2/app/deploy.sh
```

运行中的任务使用冻结源码与配置；队列完成当前任务后加载更新。

## 本地检查

```bash
python -m pip install -e . -r requirements-test.txt
python -m pytest -q
ruff check --select F,E9 msv2 tests
ruff format --check msv2 tests
bash msv2.sh doctor
```

CPU 测试覆盖真实 FFmpeg 横竖屏解码、PTS、COLMAP 文本／二进制数据关系、缩放、上传稳定性与隔离部署；`doctor` 另检查实际 CUDA 前向／反向。GPU 重建结果和 CPU 测试分别记录，不以导入或哈希检查代替完整训练。

2026-11-05 本地验证：Linux 下 76 项测试通过；真实视频抽帧及 COLMAP 恢复完成 59／59 帧配准；新入口对真实 59 视角数据完成 8,001 步验证训练，产出 658,831 个高斯，原生 PLY 回读、坐标变换及通用导出均通过。默认生产训练仍为 18,000 步。

另在全新独立 Python 环境中完成全部依赖安装、四个 CUDA 扩展从源码编译和 GPU 前向／反向检查；没有复用旧环境的包或扩展。重复部署实测通过，无重装、无重新编译。该验证使用本机已有的 CUDA Toolkit 12.8，不代表自动安装显卡驱动或验收重建质量。

上游版本、下载 SHA 和运行栈见 [upstream.lock.json](upstream.lock.json)；各组件许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
