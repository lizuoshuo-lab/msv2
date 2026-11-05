#!/usr/bin/env bash
# Pinned, isolated runtime installation. Existing interpreter mode is read-only.
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MSV2_ROOT="${MSV2_ROOT:-$HOME/msv2}"
[[ "$(uname -s)" == Linux ]] || { echo 'Use a Linux NVIDIA host or WSL2 with NVIDIA CUDA support.' >&2; exit 1; }
host_python="${MSV2_BOOTSTRAP_PYTHON:-}"
if [[ -z "$host_python" ]]; then
  for candidate in python3.10 python3.11 python3; do
    if command -v "$candidate" >/dev/null; then host_python="$(command -v "$candidate")"; break; fi
  done
fi
[[ -n "$host_python" && -x "$host_python" ]] || { echo 'Python 3.10 or 3.11 is required.' >&2; exit 1; }
mkdir -p -- "$MSV2_ROOT/downloads" "$MSV2_ROOT/vendor" "$MSV2_ROOT/weights" "$MSV2_ROOT/logs" "$MSV2_ROOT/videos" "$MSV2_ROOT/runs" "$MSV2_ROOT/queue"
existing_python="${MSV2_PYTHON:-}"
if [[ -n "$existing_python" && ! -x "$existing_python" ]]; then
  echo "MSV2_PYTHON is not executable: $existing_python" >&2; exit 1
fi
packages=()
require_tools() {
  local package="$1" executable
  shift
  for executable in "$@"; do
    if ! command -v "$executable" >/dev/null; then packages+=("$package"); return; fi
  done
}
require_tools ffmpeg ffmpeg ffprobe
require_tools colmap colmap
require_tools git git
require_tools util-linux flock
if [[ -z "$existing_python" ]]; then
  "$host_python" -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,12), "Python 3.10 or 3.11 is required"'
  require_tools build-essential g++ make
  if ! "$host_python" -c 'import venv,ensurepip' >/dev/null 2>&1; then
    packages+=("$("$host_python" -c 'import sys; print("python%d.%d-venv" % sys.version_info[:2])')")
  fi
fi
dependency_help() {
  echo 'System dependencies are missing and could not be installed automatically.' >&2
  echo 'On Ubuntu, run this command with an administrator account, then retry deployment:' >&2
  printf 'sudo apt-get update && sudo apt-get install -y' >&2
  printf ' %q' "${packages[@]}" >&2
  printf '\n' >&2
}
if [[ ${#packages[@]} -gt 0 ]]; then
  apt_prefix=()
  if [[ "$(id -u)" != 0 ]]; then
    if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then apt_prefix=(sudo -n)
    else dependency_help; exit 1
    fi
  fi
  if ! command -v apt-get >/dev/null; then dependency_help; exit 1; fi
  echo "Installing required system packages: ${packages[*]}"
  if ! "${apt_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update || \
     ! "${apt_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends "${packages[@]}"; then
    dependency_help; exit 1
  fi
fi
for executable in ffmpeg ffprobe colmap git flock; do
  command -v "$executable" >/dev/null || { echo "System dependency is still missing after installation: $executable" >&2; exit 1; }
done
exec 9>"$MSV2_ROOT/queue/install.lock"
flock -n 9 || { echo 'Another MSV2 installation is already running.' >&2; exit 1; }
if [[ ! -e "$MSV2_ROOT/config.json" ]]; then
  [[ -f "$APP_DIR/configs/default.json" ]] || { echo 'Repository configuration is missing: configs/default.json' >&2; exit 1; }
  cp -- "$APP_DIR/configs/default.json" "$MSV2_ROOT/config.json"
fi
if [[ -n "$existing_python" ]]; then
  [[ -x "$existing_python" ]] || { echo "MSV2_PYTHON is not executable: $existing_python" >&2; exit 1; }
else
  for executable in g++ make; do
    command -v "$executable" >/dev/null || { echo "Build tool missing: $executable. Install git and build-essential." >&2; exit 1; }
  done
  cuda_128() {
    [[ -x "$1/bin/nvcc" ]] && "$1/bin/nvcc" --version 2>/dev/null | grep -Eq 'release 12\.8([ ,]|$)'
  }
  if [[ -z "${CUDA_HOME:-}" ]]; then
    cuda_candidates=()
    if command -v nvcc >/dev/null; then
      cuda_candidates+=("$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")")
    fi
    cuda_candidates+=("$MSV2_ROOT/toolchains/cuda-12.8" "$HOME/.local/cuda-12.8")
    # User-installed toolkits may not be on a noninteractive SSH session's PATH.
    # These bounded globs cover common user data locations without sourcing env files.
    for nvcc_candidate in "$HOME"/.local/share/cuda*/bin/nvcc "$HOME"/.local/share/*/cuda*/bin/nvcc; do
      if [[ -x "$nvcc_candidate" ]]; then cuda_candidates+=("$(dirname -- "$(dirname -- "$nvcc_candidate")")"); fi
    done
    cuda_candidates+=(/usr/local/cuda-12.8 /usr/local/cuda)
    for cuda_candidate in "${cuda_candidates[@]}"; do
      if cuda_128 "$cuda_candidate"; then CUDA_HOME="$cuda_candidate"; break; fi
    done
  fi
  [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'CUDA Toolkit 12.8 compiler missing. Install the NVIDIA toolkit or set CUDA_HOME; the PyTorch CUDA wheel alone does not include nvcc.' >&2; exit 1; }
  cuda_128 "$CUDA_HOME" || { echo 'This pinned runtime requires CUDA Toolkit 12.8.' >&2; exit 1; }
  export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
fi
export MSV2_SOURCE_DIR="${MSV2_SOURCE_DIR:-$MSV2_ROOT/vendor/mini-splatting2}"
export MSV2_SWINIR_SOURCE="${MSV2_SWINIR_SOURCE:-$MSV2_ROOT/vendor/SwinIR}"
export MSV2_SWINIR_WEIGHT="${MSV2_SWINIR_WEIGHT:-$MSV2_ROOT/weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth}"
"$host_python" - "$APP_DIR/upstream.lock.json" "$MSV2_ROOT" <<'PY'
import hashlib, json, os, shutil, sys, tarfile, tempfile, time, urllib.request
from pathlib import Path
lock=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
root=Path(sys.argv[2])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def fetch(url, path, expected):
    if path.is_file() and digest(path)==expected:return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'msv2-deployment'})
            with urllib.request.urlopen(request,timeout=60) as source,temporary.open('wb') as output:
                shutil.copyfileobj(source,output,4*1024*1024)
            if digest(temporary)!=expected:raise RuntimeError('SHA256 mismatch: '+path.name)
            temporary.replace(path)
            return
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)
def generated_metadata(name):
    # Setuptools rewrites archive-bundled metadata when building a wheel.
    # Keep every Python/CUDA/C++ source protected, including extension sources.
    return any(part.endswith('.egg-info') for part in Path(name).parts)
def source(key, variable):
    entry=lock[key];dest=Path(os.environ[variable]);receipt=dest/'.msv2-source.json'
    if dest.exists():
        if not receipt.is_file():
            raise RuntimeError('Existing source has no pinned installation receipt: '+str(dest))
        saved=json.loads(receipt.read_text())
        if saved['commit']!=entry['commit'] or saved['archive_sha256']!=entry['sha256']:
            raise RuntimeError('Existing source version differs from the lock: '+str(dest))
        ignored=sorted(set(saved.get('generated_metadata',[]))|
                       {name for name in saved['files'] if generated_metadata(name)})
        for name,expected in saved['files'].items():
            if generated_metadata(name):continue
            if not (dest/name).is_file() or digest(dest/name)!=expected:
                raise RuntimeError('Installed source was changed: '+str(dest/name))
        if any(generated_metadata(name) for name in saved['files']):
            # Upgrade an earlier receipt after validating all real sources.
            saved['files']={name:expected for name,expected in saved['files'].items()
                            if not generated_metadata(name)}
            saved['generated_metadata']=ignored
            temporary=receipt.with_suffix(receipt.suffix+'.tmp')
            temporary.write_text(json.dumps(saved,indent=2)+'\n')
            temporary.replace(receipt)
        return
    archive=root/'downloads'/(entry['directory']+'-'+entry['commit']+'.tar.gz')
    fetch(entry['archive'],archive,entry['sha256'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=dest.name+'-install-',dir=dest.parent))
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        roots={Path(member.name).parts[0] for member in members}
        if len(roots)!=1:raise RuntimeError('Unexpected archive root')
        for member in members:
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise RuntimeError('Unsupported archive member: '+member.name)
        tar.extractall(staging,members=members)
    extracted=staging/next(iter(roots));patches=[]
    if key=='mini_splatting2':
        target=extracted/'submodules/simple-knn/simple_knn.cu'
        before=digest(target);text=target.read_text()
        if '#include <cfloat>' not in text:target.write_text('#include <cfloat>\n'+text)
        patches.append(dict(path=str(target.relative_to(extracted)),before_sha256=before,after_sha256=digest(target),change='Explicit standard cfloat header for CUDA 12.8'))
    archived_files=[p for p in sorted(extracted.rglob('*')) if p.is_file()]
    files={str(p.relative_to(extracted)):digest(p) for p in archived_files
           if not generated_metadata(p.relative_to(extracted))}
    ignored=[str(p.relative_to(extracted)) for p in archived_files
             if generated_metadata(p.relative_to(extracted))]
    data=dict(repository=entry['repository'],commit=entry['commit'],archive_sha256=entry['sha256'],compatibility_patches=patches,files=files,generated_metadata=ignored)
    (extracted/'.msv2-source.json').write_text(json.dumps(data,indent=2)+'\n')
    extracted.replace(dest)
    staging.rmdir()
source('mini_splatting2','MSV2_SOURCE_DIR')
source('swinir','MSV2_SWINIR_SOURCE')
weight=lock['swinir4_weight'];fetch(weight['url'],Path(os.environ['MSV2_SWINIR_WEIGHT']),weight['sha256'])
print('MSV2_PINNED_SOURCES_READY',flush=True)
PY
if [[ -n "$existing_python" ]]; then
  export MSV2_PYTHON="$existing_python"
  echo 'Verifying the supplied interpreter without installing or upgrading packages.'
else
  "$host_python" -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,12), "Python 3.10 or 3.11 is required"'
  export MSV2_PYTHON="$MSV2_ROOT/.venv/bin/python"
  if [[ ! -x "$MSV2_PYTHON" ]]; then "$host_python" -m venv "$MSV2_ROOT/.venv"; fi
  runtime_key="$("$host_python" - "$APP_DIR" <<'PY'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);h=hashlib.sha256()
for n in ('upstream.lock.json','requirements.txt'):h.update((p/n).read_bytes())
print(h.hexdigest())
PY
)"
  installed_key="$(cat "$MSV2_ROOT/.venv/.msv2-runtime-key" 2>/dev/null || true)"
  if [[ "$installed_key" != "$runtime_key" ]]; then
    "$MSV2_PYTHON" -m pip install --upgrade 'pip==25.0.1' 'setuptools==75.8.0' 'wheel==0.45.1'
    "$MSV2_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1' 'torchvision==0.22.1'
    "$MSV2_PYTHON" -m pip install -r "$APP_DIR/requirements.txt"
    "$MSV2_PYTHON" -m pip install --no-build-isolation --no-deps --editable "$APP_DIR"
  fi
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("$MSV2_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "NVIDIA CUDA device unavailable"; print(".".join(map(str, torch.cuda.get_device_capability())))')}"
  extension_key="$runtime_key|$TORCH_CUDA_ARCH_LIST|CUDA12.8"
  installed_extension_key="$(cat "$MSV2_ROOT/.venv/.msv2-extension-key" 2>/dev/null || true)"
  if [[ "$installed_extension_key" != "$extension_key" ]]; then
    export MAX_JOBS="${MAX_JOBS:-2}"
    for extension in simple-knn diff-gaussian-rasterization diff-gaussian-rasterization_ms fused-ssim; do
      "$MSV2_PYTHON" -m pip install --no-build-isolation --no-deps "$MSV2_SOURCE_DIR/submodules/$extension" \
        >"$MSV2_ROOT/logs/build-$extension.log" 2>&1 || { echo "CUDA extension failed: $extension; see $MSV2_ROOT/logs/build-$extension.log" >&2; exit 1; }
    done
    printf '%s\n' "$extension_key" > "$MSV2_ROOT/.venv/.msv2-extension-key"
  fi
  printf '%s\n' "$runtime_key" > "$MSV2_ROOT/.venv/.msv2-runtime-key"
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MSV2_EXTENSIONS_DIR:-}"
"$MSV2_PYTHON" -s - <<'PY'
import sys,torch
assert (3,10)<=sys.version_info[:2]<(3,12),'Expected Python 3.10 or 3.11'
assert torch.__version__=='2.7.1+cu128',torch.__version__
import torchvision
assert torchvision.__version__=='0.22.1+cu128',torchvision.__version__
import numpy,PIL,scipy,open3d,plyfile,timm,cv2
from simple_knn._C import distCUDA2
import diff_gaussian_rasterization,diff_gaussian_rasterization_ms,fused_ssim
print('MSV2_RUNTIME_IMPORTS_READY',torch.__version__)
PY
runtime_tmp="$MSV2_ROOT/runtime.env.tmp"
{
  printf 'export MSV2_PYTHON=%q\n' "$MSV2_PYTHON"
  printf 'export MSV2_SOURCE_DIR=%q\n' "$MSV2_SOURCE_DIR"
  printf 'export MSV2_SWINIR_SOURCE=%q\n' "$MSV2_SWINIR_SOURCE"
  printf 'export MSV2_SWINIR_WEIGHT=%q\n' "$MSV2_SWINIR_WEIGHT"
  if [[ -n "${MSV2_EXTENSIONS_DIR:-}" ]]; then printf 'export MSV2_EXTENSIONS_DIR=%q\n' "$MSV2_EXTENSIONS_DIR"; fi
  if [[ -n "${CUDA_HOME:-}" ]]; then
    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    printf '%s\n' 'export PATH="$CUDA_HOME/bin:$PATH"' 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
  fi
} > "$runtime_tmp"
mv -f -- "$runtime_tmp" "$MSV2_ROOT/runtime.env"
echo "MSV2_RUNTIME_READY $MSV2_ROOT"
