"""
Si Wafer 나노압입 SEM 미세균열 자동 측정 — 단일 스크립트 (AI+클래식 겸용)

목표
- 다양한 SEM 조건(가속전압/전류/폴리싱 차이)에서도 균열을 안정적으로 분할하고 길이를 자동 계산
- AI(U-Net) 엔진과, 설치가 어려울 때를 위한 클래식(비학습) 엔진 둘 다 제공

주요 기능
- train        : (선택) U-Net 학습 (torch가 있을 때)
- predict      : U-Net 또는 클래식 엔진으로 균열 마스크 예측
- measure      : 단일 이미지 길이 측정 (스케일바 자동/수동)
- measure-folder: 폴더 일괄 길이 측정 + CSV/오버레이 저장

사용 예시
  # 클래식 엔진으로 빠르게 예측 → 길이
  python crack_allinone.py predict --engine classic --images dataset/images --out outputs/preds
  python crack_allinone.py measure-folder --images dataset/images --masks outputs/preds \
      --scalebar-um 10.0 --save-vis outputs/measures --csv outputs/measures/lengths.csv

  # (선택) U-Net 학습 → U-Net 예측
  python crack_allinone.py train --images dataset/images --masks dataset/masks \
      --out outputs/ckpts/unet.pth --epochs 60 --batch-size 4
  python crack_allinone.py predict --engine torch --ckpt outputs/ckpts/unet.pth \
      --images dataset/images --out outputs/preds

주의
- 스케일바 자동 검출이 실패하면 --pixel-size-um 을 직접 넣으세요.
- 마스크 파일명은 원본과 동일한 이름으로 매칭됩니다.
- 하위폴더의 이미지를 포함하려면 predict에 --recursive 옵션을 사용하세요.
"""
#python -m venv .venv     
#.\.venv\Scripts\Activate
#powershell에서 실행하세요!

import argparse
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label

# -------------------------------------------------------
# Torch (선택): 없으면 클래식 엔진만 동작
# -------------------------------------------------------
HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
except Exception:
    HAS_TORCH = False

# =======================================================
# 공통 유틸
# =======================================================
IMG_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}  # 필요시 '.bmp' 추가 가능
def imread_gray(path: str) -> np.ndarray:
    """
    견고한 단일 이미지 로더:
    - 1) OpenCV (빠름)
    - 2) tifffile (TIFF 전용, 멀티페이지 첫 페이지)
    - 3) Pillow(PIL) (LOAD_TRUNCATED_IMAGES 허용)
    - 4) imageio (마지막 폴백)
    모두 실패 시 FileNotFoundError
    """
    p = Path(path)
    suf = p.suffix.lower()

    # ---------------- 1) OpenCV 우선 시도 ----------------
    try:
        # 16-bit 대응을 위해 UNCHANGED로 읽고 나서 그레이 변환/정규화
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif np.issubdtype(img.dtype, np.floating):
                m, M = float(img.min()), float(img.max())
                if M > m:
                    img = ((img - m) * (255.0 / (M - m))).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            elif img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return img
    except Exception:
        pass

    # ---------------- 2) tifffile 시도 -------------------
    if suf in {'.tif', '.tiff'}:
        try:
            import tifffile as tiff
            # 멀티페이지면 첫 페이지만 가져오자
            with tiff.TiffFile(str(p)) as tf:
                arr = tf.pages[0].asarray()
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            if arr.dtype == np.uint16:
                arr = (arr / 256).astype(np.uint8)
            elif np.issubdtype(arr.dtype, np.floating):
                m, M = float(arr.min()), float(arr.max())
                arr = ((arr - m) * (255.0 / (M - m))).astype(np.uint8) if M > m else np.zeros_like(arr, dtype=np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        except Exception:
            # tifffile도 못 읽으면 다음 폴백으로
            pass

    # ---------------- 3) Pillow (깨진 TIFF 허용) ----------
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # 잘린 파일도 읽기 시도
        with Image.open(str(p)) as im:
            # 멀티페이지면 첫 프레임
            try:
                im.seek(0)
            except Exception:
                pass
            im = im.convert('L')
            arr = np.array(im, dtype=np.uint8)
            return arr
    except Exception:
        pass

    # ---------------- 4) imageio 최종 폴백 ---------------
    try:
        import imageio.v2 as iio
        arr = iio.imread(str(p))
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype == np.uint16:
            arr = (arr / 256).astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.floating):
            m, M = float(arr.min()), float(arr.max())
            arr = ((arr - m) * (255.0 / (M - m))).astype(np.uint8) if M > m else np.zeros_like(arr, dtype=np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    except Exception:
        pass

    # 전부 실패 시
    raise FileNotFoundError(str(p))



# ----------------- 스케일바 검출 ----------------------

def detect_scalebar_pixel_length(img_gray: np.ndarray, roi_ratio: float = 0.12) -> Optional[int]:
    H, W = img_gray.shape
    h0 = int(max(0, H * (1 - roi_ratio)))
    roi = img_gray[h0:, :]
    roi_eq = cv2.equalizeHist(roi)
    _, bw = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 21), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(1, h)
        area = w * h
        if aspect > 6 and area > 0.002 * W * roi.shape[0]:
            best = max(best, w)
    return int(best) if best > 0 else None  # BUGFIX: int(best)

# ----------------- 길이 계산 ---------------------------

def postprocess_mask(mask_bin: np.ndarray, min_area: int = 20) -> np.ndarray:
    m = (mask_bin > 0).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    lab = label(m > 0)
    if lab.max() > 0:
        cleaned = remove_small_objects(lab, min_size=max(1, int(min_area)))
        m = (cleaned > 0).astype(np.uint8) * 255
    return m

def _neighbors_8(y, x, H, W):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx

def skeleton_longest_path_length(mask_bin: np.ndarray) -> float:
    m = (mask_bin > 0).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    sk = skeletonize(m > 0).astype(np.uint8)
    ys, xs = np.where(sk > 0)
    H, W = sk.shape
    idx = -np.ones_like(sk, dtype=np.int32)
    for i, (y, x) in enumerate(zip(ys, xs)):
        idx[y, x] = i
    adj: List[List[tuple]] = [[] for _ in range(len(ys))]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for ny, nx in _neighbors_8(y, x, H, W):
            if sk[ny, nx]:
                j = idx[ny, nx]
                if j >= 0:
                    adj[i].append((j, float(np.hypot(ny - y, nx - x))))

    def farthest(src: int):
        visited, stack = set(), [(src, -1, 0.0)]
        far = (src, 0.0)
        while stack:
            u, p, d = stack.pop()
            visited.add(u)
            if d > far[1]:
                far = (u, d)
            for v, w in adj[u]:
                if v == p or v in visited:
                    continue
                stack.append((v, u, d + w))
        return far

    a, _ = farthest(0)
    _, dist = farthest(a)
    return float(dist)

def skeleton_lengths_by_component(mask_bin: np.ndarray) -> List[float]:
    lab = label((mask_bin > 0).astype(np.uint8))
    Ls: List[float] = []
    for ridx in range(1, lab.max()+1):
        comp = (lab == ridx).astype(np.uint8) * 255
        L = skeleton_longest_path_length(comp)
        if L > 0:
            Ls.append(L)
    return Ls

# ----------------- 시각화 저장 -------------------------

def overlay_and_save(image_path: str, mask_bin: np.ndarray, save_path: str,
                     crack_px: float, pixel_um: float, length_um: float):
    img = imread_gray(image_path)
    if mask_bin.shape != img.shape:
        mask_bin = cv2.resize(mask_bin, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours((mask_bin > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, cnts, -1, (0, 255, 0), 1)
    txt = f"pixels={crack_px:.1f}, px_um={pixel_um:.6f}, length={length_um:.3f} um"
    cv2.putText(color, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, color)

# =======================================================
# 클래식(비학습) 엔진 — 설치가 쉬움
# =======================================================

def classic_predict_one(gray: np.ndarray) -> np.ndarray:
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (3,3), 0)
    edges = cv2.Canny(blur, 30, 90)
    dil = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    m = postprocess_mask(dil, min_area=20)
    return m

# =======================================================
# AI(U-Net) 엔진 — torch 있을 때만
# =======================================================
if HAS_TORCH:
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.net(x)

    class UNetSmall(nn.Module):
        def __init__(self, in_ch=1, out_ch=1, base=32):
            super().__init__()
            self.inc = DoubleConv(in_ch, base)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
            self.up1 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
            self.conv1 = DoubleConv(base*8, base*4)
            self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
            self.conv2 = DoubleConv(base*4, base*2)
            self.up3 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
            self.conv3 = DoubleConv(base*2, base)
            self.outc = nn.Conv2d(base, out_ch, 1)
        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4)
            x = self.conv1(torch.cat([x, x3], dim=1))
            x = self.up2(x)
            x = self.conv2(torch.cat([x, x2], dim=1))
            x = self.up3(x)
            x = self.conv3(torch.cat([x, x1], dim=1))
            x = self.outc(x)
            return x

    class BCEDiceLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
        def forward(self, logits, targets):
            bce = self.bce(logits, targets)
            probs = torch.sigmoid(logits)
            smooth = 1.0
            num = 2 * (probs * targets).sum(dim=(2,3)) + smooth
            den = (probs + targets).sum(dim=(2,3)) + smooth
            dice = 1 - (num / den).mean()
            return bce + dice

    class SegDatasetTorch(torch.utils.data.Dataset):
        def __init__(self, img_dir: str, mask_dir: Optional[str] = None, size: int = 512):
            self.img_paths = sorted([p for p in Path(img_dir).glob('*') if p.suffix.lower() in IMG_EXT])
            self.mask_dir = Path(mask_dir) if mask_dir else None
            self.size = size
        def __len__(self):
            return len(self.img_paths)
        def __getitem__(self, idx):
            ip = self.img_paths[idx]
            img = imread_gray(str(ip))
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            if self.mask_dir is not None:
                mp = self.mask_dir / ip.name
                msk = imread_gray(str(mp))
                msk = cv2.resize(msk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                msk = (msk > 127).astype(np.float32)
            else:
                msk = np.zeros_like(img, dtype=np.float32)
            img = (img / 255.0).astype(np.float32)[None, ...]
            msk = msk[None, ...]
            return torch.from_numpy(img), torch.from_numpy(msk), str(ip)

    def torch_train(args):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ds = SegDatasetTorch(args.images, args.masks, size=args.size)
        val_len = max(1, int(len(ds) * args.val_split))
        tr_len = len(ds) - val_len
        tr_ds, va_ds = random_split(ds, [tr_len, val_len], generator=torch.Generator().manual_seed(42))
        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
        model = UNetSmall(in_ch=1, out_ch=1, base=args.base).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = BCEDiceLoss()
        best_iou, stop, patience = -1.0, 0, args.patience
        out_ckpt = Path(args.out); out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.epochs+1):
            model.train(); tr_loss = 0.0
            for img, msk, _ in tqdm(tr_dl, desc=f"Epoch {epoch}/{args.epochs}"):
                img, msk = img.to(device), msk.to(device)
                opt.zero_grad(); logits = model(img); loss = loss_fn(logits, msk); loss.backward(); opt.step()
                tr_loss += loss.item() * img.size(0)
            tr_loss /= len(tr_dl.dataset)
            model.eval(); va_loss = 0.0; va_iou = 0.0
            with torch.no_grad():
                for img, msk, _ in va_dl:
                    img, msk = img.to(device), msk.to(device)
                    logits = model(img)
                    va_loss += loss_fn(logits, msk).item() * img.size(0)
                    probs = torch.sigmoid(logits); preds = (probs > 0.5).float()
                    inter = (preds * msk).sum(dim=(2,3)); union = (preds + msk - preds*msk).sum(dim=(2,3)) + 1e-6
                    va_iou += ((inter + 1e-6) / union).mean().item() * img.size(0)
            va_loss /= len(va_dl.dataset); va_iou /= len(va_dl.dataset)
            print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_iou={va_iou:.4f}")
            if va_iou > best_iou:
                best_iou = va_iou; stop = 0
                torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch}, out_ckpt)
                print(f"  ✔ saved: {out_ckpt}")
            else:
                stop += 1
                if stop >= patience:
                    print("  Early stopping."); break

    def torch_predict(args):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model = UNetSmall(in_ch=1, out_ch=1, base=args.base)
        model.load_state_dict(ckpt['model']); model.eval(); model.to(device)
        ds = SegDatasetTorch(args.images, mask_dir=None, size=args.size)
        out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
        for img, _, p in tqdm(DataLoader(ds, batch_size=1, shuffle=False), total=len(ds), desc='predict'):
            img = img.to(device); prob = torch.sigmoid(model(img))[0,0].cpu().numpy()
            pred = (prob > args.thr).astype(np.uint8) * 255
            pred = postprocess_mask(pred, min_area=args.min_area)
            cv2.imwrite(str(out_dir / Path(p[0]).name), pred)

# =======================================================
# 측정 (단일/폴더)
# =======================================================

def measure_single(image_path: str, mask_path: str, pixel_size_um: Optional[float],
                   scalebar_um: Optional[float], bottom_roi: float = 0.12,
                   comp_mode: str = 'longest', save_vis: Optional[str] = None) -> Dict:
    img = imread_gray(image_path)
    mask = imread_gray(mask_path)
    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = postprocess_mask(mask, min_area=20)
    px_um = pixel_size_um; scalebar_px = None
    if px_um is None:
        scalebar_px = detect_scalebar_pixel_length(img, roi_ratio=bottom_roi)
        if scalebar_px is None:
            raise RuntimeError("스케일바 자동 검출 실패. --pixel-size-um 사용 또는 더 선명한 스케일바 필요")
        if scalebar_um is None:
            raise RuntimeError("--scalebar-um 값이 필요합니다 (예: 10.0)")
        px_um = float(scalebar_um) / float(scalebar_px)
    lengths_px = skeleton_lengths_by_component(mask)
    crack_px = float(np.sum(lengths_px)) if comp_mode == 'sum' else (float(np.max(lengths_px)) if lengths_px else 0.0)
    length_um = crack_px * px_um
    if save_vis is not None:
        save_path = Path(save_vis)
        if save_path.is_dir() or save_path.suffix == '':
            save_path = save_path / (Path(image_path).stem + '_overlay.png')
        overlay_and_save(image_path, mask, str(save_path), crack_px, px_um, length_um)
    return {
        'image': image_path,
        'mask': mask_path,
        'pixel_size_um': px_um,
        'scalebar_px': scalebar_px,
        'crack_length_px': crack_px,
        'crack_length_um': length_um,
        'mode': comp_mode,
    }

def measure_folder(args):
    img_dir, msk_dir = Path(args.images), Path(args.masks)
    out_csv = Path(args.csv) if args.csv else None
    vis_dir = Path(args.save_vis) if args.save_vis else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    paths = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in IMG_EXT])
    for ip in tqdm(paths, desc='measure-folder'):
        mp = msk_dir / ip.name
        if not mp.exists():
            print(f"  ! no mask: {ip.name}"); continue
        try:
            res = measure_single(str(ip), str(mp), args.pixel_size_um, args.scalebar_um,
                                 args.bottom_roi, args.comp_mode,
                                 str(vis_dir) if vis_dir is not None else None)
            rows.append(res)
        except Exception as e:
            rows.append({'image': str(ip), 'mask': str(mp), 'error': str(e)})
    if out_csv is not None:
        import csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = ['image','mask','pixel_size_um','scalebar_px','crack_length_px','crack_length_um','mode','error']
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                for k in keys: r.setdefault(k, '')
                w.writerow(r)
        print(f"✔ CSV saved: {out_csv}")

# =======================================================
# PREDICT(공통)
# =======================================================

def predict_common(args):
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    img_root = Path(args.images)
    if args.recursive:
        paths = sorted([p for p in img_root.rglob('*') if p.suffix.lower() in IMG_EXT])
    else:
        paths = sorted([p for p in img_root.glob('*') if p.suffix.lower() in IMG_EXT])

    if not paths:
        raise RuntimeError(
            f"No images found under: {img_root} "
            f"(recursive={args.recursive}). Supported: {sorted(IMG_EXT)}"
        )

    if args.engine == 'torch':
        if not HAS_TORCH:
            raise RuntimeError("torch 미설치: pip install torch torchvision 또는 --engine classic 사용")
        torch_predict(args)
        return

    # classic
    for p in tqdm(paths, desc='predict-classic'):
        img = imread_gray(str(p))
        pred = classic_predict_one(img)
        cv2.imwrite(str(out_dir / p.name), pred)

# =======================================================
# CLI
# =======================================================

def main():
    ap = argparse.ArgumentParser(description='Si wafer crack auto measurement (AI + classic)')
    sp = ap.add_subparsers(dest='cmd', required=True)

    # train (AI)
    tr = sp.add_parser('train')
    tr.add_argument('--images', required=True)
    tr.add_argument('--masks', required=True)
    tr.add_argument('--out', required=True)
    tr.add_argument('--size', type=int, default=512)
    tr.add_argument('--epochs', type=int, default=60)
    tr.add_argument('--batch-size', type=int, default=4)
    tr.add_argument('--lr', type=float, default=1e-3)
    tr.add_argument('--val-split', type=float, default=0.1)
    tr.add_argument('--base', type=int, default=32)
    tr.add_argument('--patience', type=int, default=12)

    # predict (공통)
    pr = sp.add_parser('predict')
    pr.add_argument('--images', required=True)
    pr.add_argument('--out', required=True)
    pr.add_argument('--engine', choices=['torch','classic'], default='classic')
    pr.add_argument('--ckpt', type=str, default=None)
    pr.add_argument('--size', type=int, default=512)
    pr.add_argument('--base', type=int, default=32)
    pr.add_argument('--thr', type=float, default=0.5)
    pr.add_argument('--min-area', type=int, default=20)
    pr.add_argument('--recursive', action='store_true', help='하위폴더까지 재귀적으로 이미지 탐색')

    # measure (단일)
    me = sp.add_parser('measure')
    me.add_argument('--image', required=True)
    me.add_argument('--mask', required=True)
    me.add_argument('--pixel-size-um', type=float, default=None)
    me.add_argument('--scalebar-um', type=float, default=None)
    me.add_argument('--bottom-roi', type=float, default=0.12)
    me.add_argument('--comp-mode', choices=['longest','sum'], default='longest')
    me.add_argument('--save-vis', type=str, default=None)

    # measure-folder (일괄)
    mf = sp.add_parser('measure-folder')
    mf.add_argument('--images', required=True)
    mf.add_argument('--masks', required=True)
    mf.add_argument('--pixel-size-um', type=float, default=None)
    mf.add_argument('--scalebar-um', type=float, default=None)
    mf.add_argument('--bottom-roi', type=float, default=0.12)
    mf.add_argument('--comp-mode', choices=['longest','sum'], default='longest')
    mf.add_argument('--save-vis', type=str, default=None)
    mf.add_argument('--csv', type=str, default=None)

    args = ap.parse_args()

    if args.cmd == 'train':
        if not HAS_TORCH:
            raise RuntimeError('torch 미설치: pip install torch torchvision (또는 conda install -c pytorch pytorch torchvision cpuonly)')
        torch_train(args)
    elif args.cmd == 'predict':
        predict_common(args)
    elif args.cmd == 'measure':
        res = measure_single(args.image, args.mask, args.pixel_size_um, args.scalebar_um,
                             args.bottom_roi, args.comp_mode, args.save_vis)
        L = res['crack_length_um']
        if L < 1.0:
            print(f"Crack length ≈ {L*1000:.2f} nm (pixels={res['crack_length_px']:.1f}, pixel_size={res['pixel_size_um']:.6f} um/px)")
        else:
            print(f"Crack length ≈ {L:.3f} um (pixels={res['crack_length_px']:.1f}, pixel_size={res['pixel_size_um']:.6f} um/px)")
    elif args.cmd == 'measure-folder':
        measure_folder(args)

if __name__ == '__main__':
    main()
