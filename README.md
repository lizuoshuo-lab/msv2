# msv2

将视频或标准 COLMAP 数据集重建为通用 3D Gaussian Splatting 场景。适用于静态场景、近似静止人物与移动相机的视频，不依赖 SMPL-X 或专用查看器。

## 部署与训练

在 Ubuntu 22.04、Python 3.10/3.11、NVIDIA 驱动和 CUDA Toolkit 12.8 的机器上运行：

```bash
git clone https://github.com/lizuoshuo-lab/msv2.git ~/msv2/app && bash ~/msv2/app/deploy.sh
```

出现 `MSV2_DEPLOYED` 后，将**完整上传的视频**放进 `~/msv2/videos/`，自动开始训练。支持 MP4、MOV、MKV、AVI、M4V、WebM。WinSCP 可使用临时文件名上传，完成后自动改名；未完成的 `.filepart` 文件不会进入训练。其他上传工具建议先用 `.part` 后缀，完成后再改为视频后缀。

源码、CUDA 扩展、公开 SwinIR 与二维头部解析权重由部署命令在目标机器上获取、校验并安装；视频、权重和结果都在 Git 之外。缺失的系统工具在有 root 或免密 sudo 权限时自动安装，否则命令会给出需要安装的包。部署不安装显卡驱动，不替换其他项目的 Python 环境。

查看状态：

```bash
bash ~/msv2/app/msv2.sh status
```

也可手动训练任意路径的视频，或已有的 COLMAP 数据集：

```bash
bash ~/msv2/app/msv2.sh train /path/to/video.mp4
bash ~/msv2/app/msv2.sh train /path/to/colmap-dataset --scales 1,2,4
```

只提取头部像素时运行下面的 CPU 命令，不会启动相机恢复或训练。输入可为原图目录、`images/` 或准备好的数据集；输出必须是输入目录之外的新目录。每张图输出原尺寸二值 mask，每个头部输出保留原始 RGB 的透明裁剪 PNG，并记录原图／解码像素 SHA、bbox 和资产版本；未找到头部时输出零 mask，不生成裁剪图。

```bash
bash ~/msv2/app/msv2.sh extract-head /path/to/images --output /path/to/new-head-pixels
```

COLMAP 目录应包含 `images/` 和 `sparse/0/`，支持文本或二进制模型，以及已去畸变的 `PINHOLE`、`SIMPLE_PINHOLE` 相机。包含畸变参数的模型需先通过 COLMAP `image_undistorter` 转为标准输入。导入时保留相机、稀疏点和观测关系，统一图像名，避免多相机文件重名。

## 默认流程

1. 视频按真实 PTS 以 3 fps 选择最近源帧，去除重复选择；保留完整画面和显示方向，支持横屏、竖屏及旋转元数据。
2. COLMAP 从图像恢复相机和稀疏点；必须全部采样帧注册成功。不用人体姿态推测相机，不插值补位姿。
3. 默认分别训练原始完整图、Lanczos 插值两倍和 SwinIR 超分四倍，人物与背景一同按倍率放大。每条输入路线重新初始化高斯与优化器。
4. 图像尺寸与 `fx/fy/cx/cy` 同比例缩放，保留原相机外参；所有倍率使用同一组源帧和训练／留出划分。
5. 使用固定上游 MSV2 的 `config/fast`，`outdoor`，种子 0，18,000 步。保留原生 3,000 步半尺寸预热、2,000 步深度重初始化和 3,000／8,000 步简化；不额外限制点数，不在训练后删点。
6. 核验原生 PLY 回读、坐标变换后的协方差／SH／渲染一致性，导出通用文件，并在原图分辨率下比较同一组视角。

预热发生在原始／放大输入准备完成后；没有“先将原图缩小，再做两倍或四倍放大”的输入步骤。二维解析器内部将定位区域缩放至512像素以预测 mask，只有 mask 按最近邻映回原图；送入 SwinIR 的 RGB 始终重新从原始像素裁取。帧数、分辨率和相机参数均来自各自数据集，不固定为某个视频或 59 帧。

预检核对各输入视图的像素与相机投影，反向检查遵循原生首阶段的半尺寸预热。预检通过不代表完整训练已完成，训练容量与效果仍需本地实际验证。

`sr_region` 默认为 `full`。设置 `"sr_region": "face"` 时采用母图1倍＋面部子图4倍联合路线：

```bash
bash ~/msv2/app/msv2.sh train /path/to/video.mp4 --config ~/msv2/app/configs/face.json
```

1. 用现有二维解析器取得面部 mask，从原图读取包围盒及32像素自然上下文。
2. 母图保留原始尺寸和 PNG 字节，只对面部包围盒及自然上下文做 SwinIR 四倍超分。子图只保存带至少8个原生像素上下文的 ROI，向外对齐8像素网格，不生成完整四倍训练图；源图右下边界的补齐像素不参与子图损失。
3. mask 外使用母图原始1倍监督，mask 内使用子图4倍监督；子图 mask 外不计子图损失，羽化仅向 mask 内部。两部分在同一次优化器更新中共同拟合。没有面部子图的源帧按原始母图1倍训练。
4. 每个源帧仍是一个随机采样组，一套原始母相机；子图只作为该源帧的辅助数据，不加入 COLMAP 或独立随机采样池。子图校准为 `Kchild = S4 × T(-ROI原点) × Kparent`，外参不变；实际光栅化保留母相机的完整 FoV 和投影，并使用对应 ROI viewport，不能用普通裁剪相机替代。

面部 mask 包含皮肤、五官和耳部，排除头发、帽子、颈部与衣物。独立的 `extract-head` 命令仍输出包含头发和帽子的头部像素。此路线的 mask 同时限定子图监督范围与母图监督范围，联合损失和 ROI 渲染域单独记录；高斯模型与优化器采用原生实现。预热期间母图按实际尺寸缩放，子图 ROI 与内参同步映射，保留完整 FoV。历史整幅四倍面部合成不是这条母子图路线。

配置在 `~/msv2/config.json`，首次部署从 [默认配置](configs/default.json) 创建，更新时保留。可修改 `sample_fps`、`scales`、`importance_metric` 等；`frame_indices` 可指定递增、唯一的原视频帧号以精确复现某个数据集。训练预算需至少 8,001 步，保证完整走过原生简化阶段；默认是 18,000。完整选项见 [配置定义](msv2/config.py)。修改配置后下一项任务会采用新配置，已有任务保持冻结配置。

## 结果和日志

每次训练位于 `~/msv2/runs/<名称>/`：

- `state.json`：本轮阶段、错误和已完成倍率。
- `logs/`：抽帧、相机、预检、训练和导出的逐阶段日志；预处理详细日志另在 `prepared/logs/`。
- `swinir4x/superresolution.json`、`swinir4x/training-inputs.json`：面部 mask 与 ROI 来源、母图字节保护、辅助子图与相机映射、逐源帧训练／留出分组。面部路线的子图及损失 mask 在 `swinir4x/children/`，不在主图像采样目录。
- `export/comparison.html`：原图与各倍率逐帧对照，留出指标统一在原图尺寸计算。
- `export/<倍率>/scene.zip`：标准高斯 `point_cloud.ply`、完整 `cameras.json`、COLMAP 文本相机、坐标变换与一致性核验记录。
- `export/<倍率>/point_cloud_world.ply`、`cameras_world.json`：原训练世界坐标文件。

面部母子图路线另导出 `paired-cameras.json`（包含在 `scene.zip`）与 `paired-cameras_world.json`，记录母子关系、源帧分组、子图 K／完整 K／viewport、相同外参及 mask 依据。主 `cameras.json` 和 COLMAP 文本只包含原始母相机，辅助子相机不计入原始拍摄视角数。训练子图 PNG 不打包进通用场景 ZIP。

主 PLY 将 COLMAP 世界坐标绕 X 轴旋转 180°，中心、完整协方差、各阶 SH 和相机同步变换；它不代表测量得到的重力方向。文件采用标准 3DGS 字段，可独立导入支持该格式的查看器。COLMAP 文本导出提供相机轨迹，不包含原始特征观测。

同一时间只运行一项任务。自动队列按视频内容、配置和实现版本去重；手动 `train` 每次创建全新运行。较大倍率显存不足时会保留已完成倍率，状态明确为 `partial`，不会偷偷缩小图像或把失败记成成功。修复输入或环境后可用手动命令重训。格式和训练检查通过不等于新视角几何质量通过。

更新：

```bash
git -C ~/msv2/app pull --ff-only && bash ~/msv2/app/deploy.sh
```

运行中的任务使用冻结源码与配置；队列完成当前任务后加载更新。

## 本地检查

新建 CPU 测试环境可先安装 CPU 版 PyTorch；已有训练环境直接沿用原 PyTorch，不需要替换 CUDA 包。

```bash
# 仅新建 CPU 测试环境需要执行这一行
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . -r requirements-test.txt
python -m pytest -q
ruff check --select F,E9 msv2 tests
ruff format --check msv2 tests
bash msv2.sh doctor
```

CPU 测试覆盖真实 FFmpeg 横竖屏解码、PTS、COLMAP 文本／二进制数据关系、缩放、上传稳定性与隔离部署；`doctor` 另检查实际 CUDA 前向／反向。GPU 重建结果和 CPU 测试分别记录，不以导入或哈希检查代替完整训练。

面部母子图的 CPU 检查覆盖母图原字节保护、ROI 像素与 mask 孔洞／边界、多面部重叠归属、奇数尺寸预热映射及辅助相机导出。CPU 数据与格式检查不代表 ROI 光栅化／梯度、完整训练或重建效果通过，这些结果另行记录。

此前整幅输入路线的本地验证（2026-11-05）：Linux 下 76 项测试通过；真实视频抽帧及 COLMAP 恢复完成 59／59 帧配准；新入口对真实 59 视角数据完成 8,001 步验证训练，产出 658,831 个高斯，原生 PLY 回读、坐标变换及通用导出均通过。默认生产训练仍为 18,000 步。这些历史记录不代表重建质量验收。

历史整图输入验证：65／65 帧完成相机恢复；原始、插值两倍与全图四倍均完成18,000步、原生 PLY 回读及通用相机导出。面部合成路线分别记录输入核查、实际训练和新视角效果；旧的整图＋独立头框视图实验不作为本路线验收。

另在全新独立 Python 环境中完成全部依赖安装、四个 CUDA 扩展从源码编译和 GPU 前向／反向检查；没有复用旧环境的包或扩展。重复部署实测通过，无重装、无重新编译。该验证使用本机已有的 CUDA Toolkit 12.8，不代表自动安装显卡驱动或验收重建质量。

上游版本、下载 SHA 和运行栈见 [upstream.lock.json](upstream.lock.json)；各组件许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
