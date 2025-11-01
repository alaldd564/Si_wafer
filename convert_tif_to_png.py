# 터미널 실행코드
# python convert_tif_to_png.py 
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageFile
import tifffile as tiff

ImageFile.LOAD_TRUNCATED_IMAGES = True

in_dir = Path("dataset/images")
out_dir = Path("dataset/images_png")
out_dir.mkdir(parents=True, exist_ok=True)

def to_u8_gray(arr):
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

for p in sorted(in_dir.glob("*.tif")) + sorted(in_dir.glob("*.tiff")):
    try:
        with tiff.TiffFile(str(p)) as tf:
            arr = tf.pages[0].asarray()
    except Exception:
        try:
            im = Image.open(str(p))
            im.seek(0)
            im = im.convert('L')
            arr = np.array(im)
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
            continue

    arr = to_u8_gray(arr)
    out = out_dir / (p.stem + ".png")
    cv2.imwrite(str(out), arr)
    print(f"[OK] {p.name} -> {out.name}")
