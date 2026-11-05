"""Same-camera, native-resolution comparison without masks or detector crops."""

from html import escape
import shutil
from types import SimpleNamespace

from .engine import LazyCamera, NAMES, save_image
from .io import now, read_json, save_json


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
    save_json(
        out / "metrics.json",
        dict(
            scale=scale,
            execution=execution,
            metrics=metrics,
            means=means,
            comparison_domain="original full RGB dimensions and cameras",
            mask_used=False,
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
        variants[name] = dict(metrics=metrics["means"], delivery=delivery)
        test = metrics["means"]["test"]
        table += (
            f"<tr><td>{names[scale]}</td><td>{delivery['points']:,}</td>"
            f"<td>{test['psnr']:.3f}</td><td>{test['ssim']:.4f}</td>"
            f'<td><a href="{name}/scene.zip">PLY 与相机</a></td></tr>'
        )
    data = dict(
        created=now(),
        frames=len(rows),
        variants=variants,
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
<h1>MSV2 · 完整画面对照</h1><p>{len(rows)} 个输入视角，{engine.iterations:,} 步，{escape(engine.config["importance_metric"])} / config/fast。完整背景参与训练，无 mask、无人物裁剪；从源图直接放大。</p>
<table><tr><th>输入</th><th>高斯数</th><th>留出 PSNR</th><th>留出 SSIM</th><th>通用导出</th></tr>{table}</table>
<p>所有指标与下方渲染都使用原始图像尺寸和同一组相机。训练完成、导出一致性通过与新视角质量分别记录。</p>{notices}
<p>视角 <select id="frame">{options}</select></p><div class="grid">{figures}</div>
<p><a href="comparison.json">完整指标</a></p>
<script>const names={__import__("json").dumps([name for name, _ in panels])};const select=document.querySelector('#frame');function update(){{for(const name of names){{const path='frames/'+select.value+'/'+name+'.png';document.querySelector('#image-'+name).src=path;document.querySelector('#link-'+name).href=path}}}}select.onchange=update;update();</script></html>"""
    (engine.record / "comparison.html").write_text(html, encoding="utf-8")
