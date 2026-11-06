"""Complete-image comparisons; regional SR never adds training cameras."""

from html import escape
import shutil
from types import SimpleNamespace

from .engine import LazyCamera, NAMES, save_image
from .io import now, read_json, save_json


def view_summary(rows):
    if not rows:
        raise ValueError("At least one complete source view is required")
    sources, names = set(), set()
    for row in rows:
        source = str(row.get("source_frame", row["frame"]))
        if source in sources or row["name"] in names:
            raise ValueError(
                "Exactly one complete training view per source frame is required"
            )
        if row.get("view_type", "full") != "full":
            raise ValueError(
                "Cropped training views are not part of full-image composition"
            )
        if row["split"] not in ("train", "test"):
            raise ValueError("Unknown source split")
        sources.add(source)
        names.add(row["name"])
    train = sum(row["split"] == "train" for row in rows)
    return dict(
        source_frames=len(rows),
        view_count=len(rows),
        full_views=len(rows),
        head_views=0,
        train_views=train,
        test_views=len(rows) - train,
        train_source_frames=train,
        test_source_frames=len(rows) - train,
        source_group_split=True,
        view_dimensions=[list(wh) for wh in sorted({tuple(row["wh"]) for row in rows})],
    )


def route_metadata(engine, scale, rows):
    counts = view_summary(rows)
    mode = engine.config.get("sr_region", "full") if scale == 4 else None
    faces, enhanced_train, enhanced_test = 0, 0, 0
    if mode == "face":
        sr = read_json(engine.root / NAMES[4] / "superresolution.json")
        if (
            sr.get("state") != "completed"
            or sr.get("view_layout") != "full4x_mask_composite"
        ):
            raise ValueError("Completed face composition provenance is required")
        items = {row["name"]: row for row in sr["records"]}
        if (
            len(items) != len(rows)
            or len(items) != len(sr["records"])
            or set(items) != {row["name"] for row in rows}
        ):
            raise ValueError(
                "Composite targets must map one-to-one to complete cameras"
            )
        for row in rows:
            count = items[row["name"]]["face_count"]
            if type(count) is not int or count < 0:
                raise ValueError("Invalid applied face count")
            faces += count
            if count:
                if row["split"] == "train":
                    enhanced_train += 1
                else:
                    enhanced_test += 1
    used = mode == "face" and faces > 0
    label = {1: "原始完整图", 2: "Lanczos 插值两倍", 4: "整幅 SwinIR 超分四倍"}[scale]
    if mode == "face":
        label = (
            "面部 mask 内4倍超分 · 整图共同训练"
            if used
            else "无面部 mask · 原图插值四倍"
        )
    return dict(
        **counts,
        label=label,
        sr_region=mode,
        input_route="full4x_mask_composite" if mode == "face" else f"full{scale}x",
        view_layout="full4x_mask_composite" if mode == "face" else "full",
        full_view_scale=scale,
        training_mask_used=False,
        training_crop_used=False,
        additional_training_views=False,
        mixed_resolution=False,
        input_composite_mask=used,
        input_face_mask_used=used,
        input_head_mask_used=False,
        input_roi_composite=used,
        full_frame_4x_generated=scale == 4,
        mask_scope="face" if mode == "face" else None,
        mask_source="existing_digital_human_face_parsing" if mode == "face" else None,
        face_regions=faces,
        face_train_frames=enhanced_train,
        face_test_frames=enhanced_test,
        outside_mask_reference="Original full-image Lanczos4"
        if mode == "face"
        else None,
    )


def evaluate(engine, scale):
    torch = engine.runtime()
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from fused_ssim import fused_ssim

    run = engine.root / NAMES[scale]
    out = engine.record / NAMES[scale]
    execution = read_json(run / "execution.json")
    assert (
        execution["state"] == "training_completed"
        and read_json(run / "ply-parity.json")["passed"]
    )
    model = GaussianModel(3)
    model.load_ply(
        str(run / f"model/point_cloud/iteration_{engine.iterations}/point_cloud.ply")
    )
    model.active_sh_degree = 3
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.zeros(3, device="cuda")
    rows = read_json(engine.root / "original/training-inputs.json")["records"]
    metrics = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            camera = LazyCamera(row, index)
            prediction = render(camera, model, pipe, background)["render"].clamp(0, 1)
            target = camera.original_image
            delta = prediction - target
            metrics.append(
                dict(
                    frame=row["frame"],
                    split=row["split"],
                    wh=row["wh"],
                    l1=float(delta.abs().mean()),
                    psnr=float(
                        -10 * torch.log10(delta.square().mean().clamp_min(1e-12))
                    ),
                    ssim=float(fused_ssim(prediction[None], target[None])),
                )
            )
            folder = engine.record / "frames" / row["frame"]
            save_image(folder / (NAMES[scale] + ".png"), prediction)
            if not (folder / "input.png").exists():
                shutil.copy2(row["native_image"], folder / "input.png")
            if index % 20 == 0:
                print("EVALUATING", scale, index + 1, len(rows), flush=True)
    means = {}
    for split in ("train", "test"):
        selected = [row for row in metrics if row["split"] == split]
        means[split] = {
            key: sum(row[key] for row in selected) / len(selected)
            for key in ("l1", "psnr", "ssim")
        }
    inputs = read_json(run / "training-inputs.json")["records"]
    provenance = route_metadata(engine, scale, inputs)
    save_json(
        out / "metrics.json",
        dict(
            scale=scale,
            execution=execution,
            metrics=metrics,
            means=means,
            comparison_domain="original full RGB dimensions and cameras",
            mask_used=False,
            input_provenance=provenance,
            training_mask_used=False,
            input_composite_mask=provenance["input_composite_mask"],
            input_face_mask_used=provenance["input_face_mask_used"],
            view_layout=provenance["view_layout"],
            quality_accepted=False,
        ),
    )
    for filename in (
        "preflight.json",
        "execution.json",
        "resolved-options.json",
        "progress.jsonl",
    ):
        shutil.copy2(run / filename, out / filename)
    if scale == 4:
        shutil.copy2(run / "superresolution.json", out / "superresolution.json")


def publish(engine, scales, failures):
    engine.record.mkdir(parents=True, exist_ok=True)
    rows = read_json(engine.root / "original/training-inputs.json")["records"]
    for row in rows:
        folder = engine.record / "frames" / row["frame"]
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "input.png").exists():
            shutil.copy2(row["native_image"], folder / "input.png")
    names = {1: "原始完整图", 2: "Lanczos 插值两倍", 4: "SwinIR 超分四倍"}
    variants = {}
    table = ""
    for scale in scales:
        name = NAMES[scale]
        metrics = read_json(engine.record / name / "metrics.json")
        delivery = read_json(engine.record / name / "delivery.json")
        assert delivery["parity_passed"] and delivery["cameras"] == len(rows)
        assert metrics["execution"]["state"] == "training_completed"
        inputs = read_json(engine.root / name / "training-inputs.json")["records"]
        provenance = route_metadata(engine, scale, inputs)
        assert provenance["source_frames"] == len(rows)
        if "train_frames" in metrics["execution"]:
            assert metrics["execution"]["train_frames"] == provenance["train_views"]
            assert metrics["execution"]["test_frames"] == provenance["test_views"]
        names[scale] = provenance["label"]
        variants[name] = dict(metrics=metrics["means"], delivery=delivery, **provenance)
        test = metrics["means"]["test"]
        table += (
            f"<tr><td>{names[scale]}</td><td>{delivery['points']:,}</td>"
            f"<td>{test['psnr']:.3f}</td><td>{test['ssim']:.4f}</td>"
            f'<td><a href="{name}/scene.zip">PLY 与相机</a></td></tr>'
        )
    data = dict(
        created=now(),
        frames=len(rows),
        source_frames=len(rows),
        variants=variants,
        additional_training_views=False,
        training_mask_used=False,
        sr_region=engine.config.get("sr_region", "full"),
        failures=failures,
        requested_scales=engine.config["scales"],
        completed_scales=scales,
        training_completed=len(scales) == len(engine.config["scales"]),
        quality_accepted=False,
    )
    save_json(engine.record / "comparison.json", data)
    options = "".join(
        f'<option value="{escape(row["frame"])}">{escape(row["frame"])} · '
        f"{'留出' if row['split'] == 'test' else '训练'}</option>"
        for row in rows
    )
    panels = [("input", "视频／数据集原图")] + [
        (NAMES[scale], names[scale]) for scale in scales
    ]
    figures = "".join(
        f'<figure><figcaption>{label}</figcaption><a target="_blank" id="link-{name}">'
        f'<img id="image-{name}"></a></figure>'
        for name, label in panels
    )
    notices = "".join(
        f"<p>倍率 {escape(str(scale))} 未完成：{escape(message)}</p>"
        for scale, message in failures.items()
    )
    html = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>MSV2 倍率对照</title>
<style>body{{font:16px/1.6 system-ui;margin:24px;background:#f5f6f7;color:#20242a}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}.grid{{display:grid;grid-template-columns:repeat({max(1, len(panels))},minmax(0,1fr));gap:12px}}figure{{margin:0}}img{{width:100%;height:auto}}select{{padding:8px}}a{{color:#2463a4}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>MSV2 · 完整画面对照</h1><p>{len(rows)} 个源帧、每路线 {len(rows)} 张完整训练图和相机，{engine.iterations:,} 步，{escape(engine.config["importance_metric"])} / config/fast。</p><p>面部路线按二维 mask 提取原始包围盒与上下文做四倍超分，再贴回同一完整图。mask 外使用原图 Lanczos 四倍对齐像素网格，保留原始内容；每帧只有一套整图相机，统一 K×4、外参不变。人物与背景共同拟合；原生模型、损失与优化器保持原样。无面部时整图为原图插值四倍。</p>
<table><tr><th>输入</th><th>高斯数</th><th>留出 PSNR</th><th>留出 SSIM</th><th>通用导出</th></tr>{table}</table>
<p>所有指标与下方渲染都使用原始图像尺寸和同一组相机。训练完成、导出一致性通过与新视角质量分别记录。</p>{notices}
<p>视角 <select id="frame">{options}</select></p><div class="grid">{figures}</div>
<p><a href="comparison.json">完整指标</a></p>
<script>const names={__import__("json").dumps([name for name, _ in panels])};const select=document.querySelector('#frame');function update(){{for(const name of names){{const path='frames/'+select.value+'/'+name+'.png';document.querySelector('#image-'+name).src=path;document.querySelector('#link-'+name).href=path}}}}select.onchange=update;update();</script></html>"""
    (engine.record / "comparison.html").write_text(html, encoding="utf-8")
