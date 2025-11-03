# measure_cracks.py  (Raycasting + peak picking, length cap)
# -----------------------------------------------------------------------------
# 예시:
#   python -u measure_cracks.py --image sem_03.png --scale_um 10 --bar_px 87 --max_um 12 --out outputs/annotated.png
#   python -u measure_cracks.py --image sem_03.png --um_per_px 0.11494 --max_um 12 --out outputs/annotated.png
#   python -u measure_cracks.py --image sem_03.png --scale_um 10 --bar_px 87 --center "620,470" --out outputs/annotated.png
# -----------------------------------------------------------------------------

import os, cv2, numpy as np
from math import atan2, degrees, cos, sin, hypot
from collections import Counter

# ---------- I/O ----------
def imread_unicode(path):
    path = os.path.abspath(path)
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.imwrite(path, img)

# ---------- Scale bar detection (robust) ----------
def auto_scale_bar_width_px(gray, bottom_frac=0.25):
    h, w = gray.shape
    y0 = int(h*(1-bottom_frac))
    crop = gray[y0:, :]
    crop_eq = cv2.equalizeHist(crop)
    _, th = cv2.threshold(crop_eq, 220, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (25,5)), 2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_score = None, -1
    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        y_abs = y + y0
        area = wc*hc
        aspect = wc/max(hc,1)
        if (y_abs > h*0.82 and hc < 0.08*h and aspect > 12 and wc > 0.10*w and area > (w*h)*0.0005):
            score = wc*aspect
            if score > best_score:
                best_score = score
                best = (x, y_abs, wc, hc)
    if not best:
        # fallback: hough long horizontal
        y0 = int(h*(1-bottom_frac))
        crop = gray[y0:, :]
        edges = cv2.Canny(cv2.equalizeHist(crop), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, int(0.08*w), 8)
        if lines is None: raise RuntimeError("스케일바 자동 검출 실패")
        best_len, best = -1, None
        for x1,y1,x2,y2 in lines[:,0,:]:
            ang = abs(degrees(np.arctan2(y2-y1, x2-x1))); ang = min(ang, 180-ang)
            L = hypot(x2-x1, y2-y1); ymid = (y1+y2)/2 + y0
            if ang < 5 and ymid > h*0.82 and L > best_len:
                best_len, best = L, (min(x1,x2), min(y1,y2)+y0, abs(x2-x1), max(abs(y2-y1),3))
        if best is None: raise RuntimeError("스케일바 자동 검출 실패")
        return int(round(best_len)), best
    x,y,wc,hc = best
    return wc, (x,y,wc,hc)

# ---------- Center estimation via line-intersection votes ----------
def lines_from_hough(gray):
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),1.2), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=45, minLineLength=25, maxLineGap=20)
    return ([] if lines is None else [l[0] for l in lines])

def line_params(seg):
    x1,y1,x2,y2 = seg; return (y1-y2, x2-x1, x1*y2 - x2*y1)

def intersection(l1, l2):
    a1,b1,c1 = line_params(l1); a2,b2,c2 = line_params(l2)
    det = a1*b2 - a2*b1
    if abs(det) < 1e-6: return None
    x = (b1*c2 - b2*c1)/det; y = (c1*a2 - c2*a1)/det
    return (x,y)

def estimate_center_via_votes(gray):
    h,w = gray.shape
    lines = lines_from_hough(gray)
    votes = []
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            pt = intersection(lines[i], lines[j])
            if pt is None: continue
            x,y = pt
            if 0 <= x < w and 0 <= y < h: votes.append((x,y))
    if not votes: return (w/2, h/2)
    pts = np.array(votes, np.float32)
    bins = np.round(pts/5).astype(int)
    mode, _ = Counter(map(tuple,bins)).most_common(1)[0]
    cluster = pts[(bins[:,0]==mode[0]) & (bins[:,1]==mode[1])]
    c = cluster.mean(axis=0)
    return (float(c[0]), float(c[1]))

# ---------- Raycasting & 3-peak picking ----------
def raycast_three_branches(gray, center, um_per_px, max_radius_px=120, step_deg=1, sep_deg=35):
    """
    중심에서 0..179° 모든 방향으로 1px씩 전진하여 Canny edge를 만난 '가장 먼 점'을 기록.
    각도-거리 곡선에서 비최대 억제(NMS)로 분리된 3개 피크를 선택.
    """
    cx, cy = center
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),1.2), 50, 150)

    angles = np.arange(0, 180, step_deg, dtype=np.float32)
    dist = np.zeros_like(angles)
    tip  = [(int(cx),int(cy))]*len(angles)

    for i,a in enumerate(angles):
        rad = np.deg2rad(a); dx, dy = cos(rad), sin(rad)
        last_hit = None; miss = 0
        for r in range(5, int(max_radius_px)):
            x = int(round(cx + dx*r)); y = int(round(cy + dy*r))
            if x<0 or x>=w or y<0 or y>=h: break
            if edges[y,x] > 0:
                last_hit = (x,y); miss = 0
            else:
                miss += 1
                if miss > 4: break
        if last_hit is not None:
            dist[i] = hypot(last_hit[0]-cx, last_hit[1]-cy)
            tip[i]  = last_hit

    # NMS로 3개 피크
    picked = []
    arr = dist.copy()
    win = max(1, int(sep_deg/step_deg))
    for _ in range(3):
        k = int(np.argmax(arr))
        if arr[k] <= 0: break
        picked.append(k)
        L = max(0, k-win); R = min(len(arr), k+win+1)
        arr[L:R] = 0

    # 결과
    lens_px = [float(dist[k]) for k in picked]
    lens_um = [d*um_per_px for d in lens_px]
    tips    = [tip[k] for k in picked]
    angs    = [float(angles[k]) for k in picked]
    return lens_px, lens_um, tips, angs

# ---------- Main ----------
def run(image_path, scale_um=None, out_path=None,
        bar_px_override=None, um_per_px_override=None,
        center_override=None, max_um=None, max_radius_px=None):
    img = imread_unicode(image_path); assert img is not None, f"이미지를 못 읽었습니다: {image_path}"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # scale
    if um_per_px_override is not None:
        um_per_px = float(um_per_px_override); bar_px = None; bar_bbox = None
    elif bar_px_override is not None:
        assert scale_um is not None, "--bar_px 사용 시 --scale_um 필요"
        bar_px = float(bar_px_override); bar_bbox = None; um_per_px = float(scale_um)/bar_px
    else:
        bar_px, bar_bbox = auto_scale_bar_width_px(gray)
        assert scale_um is not None, "자동 스케일바 사용 시 --scale_um 필요"
        um_per_px = float(scale_um)/float(bar_px)

    # center
    if center_override is not None:
        cx, cy = map(float, center_override.split(","))
        center = (cx, cy)
    else:
        center = estimate_center_via_votes(gray)

    # 최대 반경 결정 (우선순위: max_um > max_radius_px > 기본 120px)
    if max_um is not None:
        max_radius_px_eff = float(max_um)/um_per_px
    elif max_radius_px is not None:
        max_radius_px_eff = float(max_radius_px)
    else:
        max_radius_px_eff = 120.0

    # raycasting으로 3개 길이
    branch_px, branch_um, tips, angs = raycast_three_branches(
        gray, center, um_per_px, max_radius_px=max_radius_px_eff, step_deg=1, sep_deg=35
    )

    # 시각화
    vis = img.copy()
    cv2.circle(vis, (int(center[0]), int(center[1])), 6, (0,255,0), -1)
    if bar_px_override is not None:
        txt = f"scale {um_per_px:.5f} um/px  (bar {int(bar_px)} px, {scale_um} um)"
        cv2.putText(vis, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    elif bar_bbox is not None:
        x,y,wc,hc = bar_bbox
        cv2.rectangle(vis, (x,y), (x+wc,y+hc), (255,0,0), 2)
        cv2.putText(vis, f"bar {int(float(scale_um)/um_per_px)} px = {scale_um} um",
                    (x, max(15,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 1, cv2.LINE_AA)
    else:
        cv2.putText(vis, f"scale {um_per_px:.5f} um/px", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)

    # 브랜치 표시/라벨
    colors = [(0,0,255),(0,255,255),(255,0,255)]
    for i,(px,um,tp,ang) in enumerate(zip(branch_px, branch_um, tips, angs), 1):
        cv2.line(vis, (int(center[0]), int(center[1])), (int(tp[0]), int(tp[1])), (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{i}: {um:.3f} um @ {ang:.0f}°",
                    (int(tp[0])+5, int(tp[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)

    if out_path is None:
        base,_ = os.path.splitext(os.path.abspath(image_path))
        out_path = base + "_annotated.png"
    assert imwrite_unicode(out_path, vis), f"이미지 저장 실패: {out_path}"

    # 콘솔 출력
    print(f"Scale: {scale_um if scale_um is not None else '-'} um / "
          f"{'—' if bar_px_override is None else int(bar_px_override)} px  ->  {um_per_px:.5f} um/px")
    if max_um is not None:
        print(f"(length cap) max_um = {max_um} um  -> max_radius_px ≈ {max_radius_px_eff:.1f} px")
    elif max_radius_px is not None:
        print(f"(length cap) max_radius_px = {max_radius_px_eff:.1f} px")

    for i,(px,um,ang) in enumerate(zip(branch_px, branch_um, angs), 1):
        print(f"Branch {i}: {px:.1f} px = {um:.3f} um  (angle≈{ang:.0f}°)")
    print(f"Annotated saved: {out_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="SEM 이미지 경로")
    p.add_argument("--scale_um", type=float, default=None, help="스케일바 실제 길이(µm)")
    p.add_argument("--out", default=None, help="주석 결과 저장 경로")
    p.add_argument("--bar_px", type=float, default=None, help="스케일바 폭(px) 수동 지정")
    p.add_argument("--um_per_px", type=float, default=None, help="픽셀당 µm 직접 지정")
    p.add_argument("--center", type=str, default=None, help='중심 고정 "x,y" (예: "620,470")')
    p.add_argument("--max_um", type=float, default=None, help="최대 반경(µm) 제한")
    p.add_argument("--max_radius_px", type=float, default=None, help="최대 반경(px) 제한")
    args = p.parse_args()

    run(
        image_path=args.image,
        scale_um=args.scale_um,
        out_path=args.out,
        bar_px_override=args.bar_px,
        um_per_px_override=args.um_per_px,
        center_override=args.center,
        max_um=args.max_um,
        max_radius_px=args.max_radius_px
    )
