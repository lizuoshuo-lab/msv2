"""Source-frame comparisons with explicit mixed-resolution view provenance."""

from html import escape
from pathlib import Path
import shutil
from types import SimpleNamespace

from .engine import LazyCamera, NAMES, save_image
from .io import now, read_json, save_json

NATIVE_FOOTPRINT_LIMITATION = (
    "Crop intrinsics preserve projected pixel centers and the original camera pose. "
    "The unchanged native rasterizer uses a symmetric FoV covariance clamp; "
    "direct head renders are not guaranteed pixel-identical to full4 renders cropped afterward."
)
HEAD_MASK_SOURCE = "existing_digital_human_face_parsing"


def sr_details(engine):
    path = engine.root / "swinir4x/superresolution.json"
    return read_json(path) if path.exists() else {}


def labels(engine, sr, counts=None, capacity=None):
    names = {1: "原始完整图", 2: "Lanczos 插值两倍", 4: "整幅 SwinIR 超分四倍"}
    if engine.config.get("sr_region", "full") == "head":
        names[4] = "头部 mask 内 SwinIR 4倍 + 整图1倍"
        if (
            counts is not None
            and counts["head_views"] == 0
            and capacity
            and capacity["skipped_head_view_count"]
        ):
            names[4] = "头框因初始深度配额跳过 · 仅完整原图1倍"
        elif (counts is not None and counts["head_views"] == 0) or (
            counts is None
            and sr.get("state") == "completed"
            and not any(row.get("heads") for row in sr.get("records", []))
        ):
            names[4] = "未提取有效头部 mask · 仅完整原图1倍（no_head）"
        elif counts is not None and counts["head_train_views"] == 0:
            names[4] = "头部 mask 仅留出评估 · 训练仅完整原图1倍"
    return names


def view_summary(rows):
    groups = {}
    for row in rows:
        source = str(row.get("source_frame", row["frame"]))
        split = row["split"]
        if source in groups and groups[source] != split:
            raise ValueError(
                "Views from one source frame must share a split: " + source
            )
        groups[source] = split
    return dict(
        source_frames=len(groups),
        view_count=len(rows),
        full_views=sum(row.get("view_type", "full") == "full" for row in rows),
        head_views=sum(row.get("view_type") == "head" for row in rows),
        head_train_views=sum(
            row.get("view_type") == "head" and row["split"] == "train" for row in rows
        ),
        head_test_views=sum(
            row.get("view_type") == "head" and row["split"] == "test" for row in rows
        ),
        train_views=sum(row["split"] == "train" for row in rows),
        test_views=sum(row["split"] == "test" for row in rows),
        train_source_frames=sum(split == "train" for split in groups.values()),
        test_source_frames=sum(split == "test" for split in groups.values()),
        source_group_split=True,
        view_dimensions=sorted({tuple(row["wh"]) for row in rows}),
    )


def head_capacity(metadata, sr, counts):
    generated = sum(len(row.get("heads", [])) for row in sr.get("records", []))
    skipped = metadata.get("skipped_head_views", [])
    if not isinstance(skipped, list):
        raise ValueError("skipped_head_views must retain per-view reasons")
    if metadata.get("generated_head_view_count", generated) != generated:
        raise ValueError("Generated head-view counts disagree with SR provenance")
    if counts["head_views"] + len(skipped) != generated:
        raise ValueError(
            "Every generated head view must be accepted or explicitly skipped"
        )
    reasons = {}
    for item in skipped:
        reason = str(item["reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    return dict(
        generated_head_view_count=generated,
        skipped_head_view_count=len(skipped),
        skipped_head_views=skipped,
        skipped_head_reasons=reasons,
        initial_depth_quota=metadata.get("initial_depth_quota"),
        initial_quota_is_not_dynamic_training_guarantee=True,
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
    counts = view_summary(read_json(run / "training-inputs.json")["records"])
    head_mask_used = (
        scale == 4
        and engine.config.get("sr_region", "full") == "head"
        and counts["head_views"] > 0
    )
    save_json(
        out / "metrics.json",
        dict(
            scale=scale,
            execution=execution,
            metrics=metrics,
            means=means,
            comparison_domain="original full RGB dimensions and cameras",
            mask_used=False,
            training_mask_used=False,
            input_composite_mask=head_mask_used,
            input_head_mask_used=head_mask_used,
            input_roi_composite=head_mask_used,
            full_frame_4x_generated=scale == 4
            and engine.config.get("sr_region", "full") == "full",
            head_mask_source=HEAD_MASK_SOURCE if head_mask_used else None,
            sr_region=engine.config.get("sr_region", "full") if scale == 4 else None,
            source_frames=len(rows),
            training_views=counts,
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
    sr = sr_details(engine)
    names = labels(engine, sr)
    mode = engine.config.get("sr_region", "full")
    variants, table = {}, ""
    for scale in scales:
        name = NAMES[scale]
        metrics = read_json(engine.record / name / "metrics.json")
        delivery = read_json(engine.record / name / "delivery.json")
        metadata = read_json(engine.root / name / "training-inputs.json")
        training_rows = metadata["records"]
        counts = view_summary(training_rows)
        assert counts["source_frames"] == len(rows)
        assert delivery["parity_passed"] and delivery["cameras"] == counts["view_count"]
        assert metrics["execution"]["state"] == "training_completed"
        if "train_frames" in metrics["execution"]:
            assert metrics["execution"]["train_frames"] == counts["train_views"]
            assert metrics["execution"]["test_frames"] == counts["test_views"]
        head_mode = scale == 4 and mode == "head"
        capacity = head_capacity(metadata, sr, counts) if head_mode else {}
        head_mask_used = head_mode and counts["head_views"] > 0
        if head_mask_used and sr.get("head_mask_source") != HEAD_MASK_SOURCE:
            raise ValueError(
                "Head inputs must retain the existing 2D pixel-parser source"
            )
        if head_mode:
            names[4] = labels(engine, sr, counts, capacity)[4]
        route = (
            (
                "full1x_head4x"
                if counts["head_train_views"]
                else "full1x_head4x_holdout_only"
                if counts["head_views"]
                else "full1x_all_heads_capacity_skipped"
                if capacity["skipped_head_view_count"]
                else "full1x_no_head"
            )
            if head_mode
            else f"full{scale}x"
        )
        variants[name] = dict(
            metrics=metrics["means"],
            delivery=delivery,
            label=names[scale],
            sr_region=mode if scale == 4 else None,
            input_route=route,
            training_mask_used=False,
            input_composite_mask=head_mask_used,
            input_head_mask_used=head_mask_used,
            input_roi_composite=head_mask_used,
            full_frame_4x_generated=scale == 4 and not head_mode,
            head_mask_source=sr.get("head_mask_source") if head_mode else None,
            full_view_scale=1 if head_mode else scale,
            head_view_scale=4 if counts["head_views"] else None,
            head_sr_training_used=counts["head_train_views"] > 0,
            **counts,
            **capacity,
        )
        if head_mode and counts["head_views"]:
            variants[name]["native_footprint_limitation"] = NATIVE_FOOTPRINT_LIMITATION
            variants[name]["parser_detector_assets"] = sr.get("parser_detector_assets")
        test = metrics["means"]["test"]
        table += (
            f"<tr><td>{names[scale]}</td><td>{counts['source_frames']}</td>"
            f"<td>{counts['full_views']} / {counts['head_views']}</td>"
            f"<td>{counts['train_views']} / {counts['test_views']}</td><td>{delivery['points']:,}</td>"
            f"<td>{test['psnr']:.3f}</td><td>{test['ssim']:.4f}</td>"
            f'<td><a href="{name}/scene.zip">PLY 与全部相机</a></td></tr>'
        )
    data = dict(
        created=now(),
        frames=len(rows),
        source_frames=len(rows),
        variants=variants,
        failures=failures,
        requested_scales=engine.config["scales"],
        completed_scales=scales,
        training_completed=len(scales) == len(engine.config["scales"]),
        quality_accepted=False,
        sr_region=mode,
        source_group_split=True,
        training_mask_used=False,
        input_composite_mask=any(
            row["input_composite_mask"] for row in variants.values()
        ),
        input_head_mask_used=any(
            row["input_head_mask_used"] for row in variants.values()
        ),
        input_roi_composite=any(
            row["input_roi_composite"] for row in variants.values()
        ),
        full_frame_4x_generated=any(
            row["full_frame_4x_generated"] for row in variants.values()
        ),
    )
    save_json(engine.record / "comparison.json", data)
    options = "".join(
        f'<option value="{escape(row["frame"])}">{escape(row["frame"])} · '
        f"{'同源全部留出' if row['split'] == 'test' else '训练源帧'}</option>"
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
        f"<p>输入路线 {escape(str(scale))} 未完成：{escape(message)}</p>"
        for scale, message in failures.items()
    )
    head_counts, generated_counts, sr_notice = {}, {}, ""
    if 4 in engine.config["scales"]:
        if mode == "head":
            generated_counts = {
                Path(row["name"]).stem: len(row.get("heads", []))
                for row in sr.get("records", [])
            }
            input_path = engine.root / "swinir4x/training-inputs.json"
            current = read_json(input_path) if input_path.exists() else {}
            head_counts = (
                {row["frame"]: 0 for row in rows} if current.get("records") else {}
            )
            for row in current.get("records", []):
                if row.get("view_type") == "head":
                    source = str(row["source_frame"])
                    head_counts[source] = head_counts.get(source, 0) + 1
            skipped = current.get("skipped_head_views", [])
            accepted = sum(head_counts.values())
            train_heads = sum(
                row.get("view_type") == "head" and row["split"] == "train"
                for row in current.get("records", [])
            )
            generated = sum(generated_counts.values())
            reasons = sorted({str(item["reason"]) for item in skipped})
            reason_labels = {
                "below_initial_native_depth_quota": "头框像素少于初始深度采样配额"
            }
            reason_text = ", ".join(
                reason_labels.get(reason, reason) for reason in reasons
            )
            sr_notice = (
                "复用既有数字人工程的二维头部像素解析，以 mask 提取脸、耳、头发和帽子；不做三维头模、拟合或投影。"
                "每个源帧保留完整原图1倍，有效头部 ROI 另作4倍输入，只有 mask 内保留 SwinIR 结果；没有整幅4倍画布，框外没有放大或合成。"
                f"已生成 {generated} 个头框SR，实际纳入输入 {accepted} 个（训练 {train_heads} / 留出 {accepted - train_heads}），容量跳过 {len(skipped)} 个"
                + (f"（{escape(reason_text)}）" if reasons else "")
                + "。mask 只用于输入 ROI 的超分合成，不参与训练损失。解析器的512像素输入只预测 mask；超分 RGB 重新从原图裁取。有提取结果但因容量跳过的头框不记为 no_head，也不算作模型SR增强。所有完整原图保留；同源输入共同训练或共同留出。初始配额检查不保证第2000步动态点数的采样容量。"
            )
            notices += "<p>头框内参采用 fx/fy×4、cx/cy=(原主点−头框原点)×4，外参不变。原生光栅器保留视场协方差截断近似，直接头框渲染不保证与整图4倍渲染后裁剪逐像素相同；本地实测与质量验收单独记录。</p>"
        else:
            sr_notice = "四倍对照使用整幅 SwinIR 输入，完整画面与内参同时放大四倍。"
    html = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>MSV2 输入路线对照</title>
<style>body{{font:16px/1.6 system-ui;margin:24px;background:#f5f6f7;color:#20242a}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}.grid{{display:grid;grid-template-columns:repeat({max(1, len(panels))},minmax(0,1fr));gap:12px}}figure{{margin:0}}img{{width:100%;height:auto}}select{{padding:8px}}a{{color:#2463a4}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>MSV2 · 输入路线对照</h1><p>{len(rows)} 个真实源帧，{engine.iterations:,} 步，{escape(engine.config["importance_metric"])} / config/fast。多分辨率视图属于输入数据路线；原生模型、损失、优化器与训练日程保持原样。</p><p>{sr_notice}</p>
<table><tr><th>输入</th><th>源帧</th><th>完整图 / 头框视图</th><th>训练 / 留出视图</th><th>高斯数</th><th>留出 PSNR</th><th>留出 SSIM</th><th>通用导出</th></tr>{table}</table>
<p>所有整体指标和下方渲染都使用原始完整图尺寸与同一组原相机。源帧数与展开视图数分别记录；同源留出避免头框视图泄漏。训练完成、导出一致性与新视角质量分别记录。</p>{notices}
<p>源帧 <select id="frame">{options}</select></p><p id="sr-frame-note"></p><div class="grid">{figures}</div>
<p><a href="comparison.json">完整指标与视图统计</a></p>
<script>const names={__import__("json").dumps([name for name, _ in panels])},headCounts={__import__("json").dumps(head_counts)},generatedCounts={__import__("json").dumps(generated_counts)};const select=document.querySelector('#frame');function update(){{for(const name of names){{const path='frames/'+select.value+'/'+name+'.png';document.querySelector('#image-'+name).src=path;document.querySelector('#link-'+name).href=path}}const n=headCounts[select.value],g=generatedCounts[select.value];document.querySelector('#sr-frame-note').textContent=n===0&&g>0?'本源帧有头框SR生成，但容量跳过全部头框；模型输入只有完整原图1倍。':n===0&&g===0?'本源帧 no_head：只保留完整原图1倍。':n>0?'本源帧：完整原图1倍 + '+n+' 个有效头框4倍输入，容量跳过 '+(g-n)+' 个；下图为共同原尺寸评估。':''}}select.onchange=update;update();</script></html>"""
    (engine.record / "comparison.html").write_text(html, encoding="utf-8")
