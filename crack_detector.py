"""
균열 길이 자동 측정 시스템
- PNG 이미지를 입력으로 사용
- 자동으로 스케일바 검출 및 균열 길이 측정
- JSON ground truth와 비교
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from math import atan2, degrees, cos, sin, hypot
from collections import Counter

# ==================== I/O 함수 ====================
def imread_unicode(path):
    """유니코드 경로 지원 이미지 읽기"""
    path = os.path.abspath(path)
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    """유니코드 경로 지원 이미지 저장"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _, ext = os.path.splitext(path)
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(path)
        return True
    return False

# ==================== 스케일바 자동 검출 ====================
def auto_detect_scale_bar(gray, bottom_frac=0.25):
    """
    이미지 하단에서 스케일바 자동 검출
    Returns: (bar_width_px, (x, y, w, h))
    """
    h, w = gray.shape
    y0 = int(h * (1 - bottom_frac))
    crop = gray[y0:, :]
    
    # 대비 향상
    crop_eq = cv2.equalizeHist(crop)
    
    # 이진화 (밝은 영역 추출)
    _, th = cv2.threshold(crop_eq, 220, 255, cv2.THRESH_BINARY)
    
    # 형태학 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_bbox = None
    best_score = -1
    
    for cnt in contours:
        x, y, wc, hc = cv2.boundingRect(cnt)
        y_abs = y + y0
        area = wc * hc
        aspect = wc / max(hc, 1)
        
        # 스케일바 조건: 하단 위치, 낮은 높이, 높은 종횡비, 충분한 너비
        if (y_abs > h * 0.82 and 
            hc < 0.08 * h and 
            aspect > 12 and 
            wc > 0.10 * w and 
            area > (w * h) * 0.0005):
            
            score = wc * aspect
            if score > best_score:
                best_score = score
                best_bbox = (x, y_abs, wc, hc)
    
    if best_bbox is None:
        # Fallback: Hough 변환으로 긴 수평선 검출
        crop = gray[y0:, :]
        edges = cv2.Canny(cv2.equalizeHist(crop), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, int(0.08*w), 8)
        
        if lines is None:
            raise RuntimeError("스케일바를 자동으로 검출하지 못했습니다.")
        
        best_len = -1
        best_bbox = None
        
        for x1, y1, x2, y2 in lines[:, 0, :]:
            ang = abs(degrees(np.arctan2(y2-y1, x2-x1)))
            ang = min(ang, 180-ang)
            L = hypot(x2-x1, y2-y1)
            ymid = (y1+y2)/2 + y0
            
            if ang < 5 and ymid > h*0.82 and L > best_len:
                best_len = L
                best_bbox = (min(x1,x2), min(y1,y2)+y0, abs(x2-x1), max(abs(y2-y1),3))
        
        if best_bbox is None:
            raise RuntimeError("스케일바를 자동으로 검출하지 못했습니다.")
        
        return int(round(best_len)), best_bbox
    
    x, y, wc, hc = best_bbox
    return wc, best_bbox

# ==================== 중심점 추정 ====================
def estimate_crack_center(gray):
    """
    선분 교차점 투표로 균열 중심 추정
    """
    h, w = gray.shape
    
    # Canny 엣지 검출
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 1.2), 50, 150)
    
    # Hough 변환으로 선분 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=45, 
                            minLineLength=25, maxLineGap=20)
    
    if lines is None:
        return (w/2, h/2)
    
    lines = [l[0] for l in lines]
    
    # 선분 교차점 계산
    votes = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is None:
                continue
            x, y = pt
            if 0 <= x < w and 0 <= y < h:
                votes.append((x, y))
    
    if not votes:
        return (w/2, h/2)
    
    # 투표 기반 중심점 추정 (클러스터링)
    pts = np.array(votes, np.float32)
    bins = np.round(pts / 5).astype(int)
    mode, _ = Counter(map(tuple, bins)).most_common(1)[0]
    
    cluster = pts[(bins[:, 0] == mode[0]) & (bins[:, 1] == mode[1])]
    center = cluster.mean(axis=0)
    
    return (float(center[0]), float(center[1]))

def line_intersection(seg1, seg2):
    """두 선분의 교차점 계산"""
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2
    
    a1 = y1 - y2
    b1 = x2 - x1
    c1 = x1*y2 - x2*y1
    
    a2 = y3 - y4
    b2 = x4 - x3
    c2 = x3*y4 - x4*y3
    
    det = a1*b2 - a2*b1
    if abs(det) < 1e-6:
        return None
    
    x = (b1*c2 - b2*c1) / det
    y = (c1*a2 - c2*a1) / det
    
    return (x, y)

# ==================== 레이캐스팅으로 균열 길이 측정 ====================
def raycast_measure_cracks(gray, center, um_per_px, 
                          max_radius_px=120, 
                          step_deg=1, 
                          sep_deg=35):
    """
    중심에서 방사형으로 레이캐스팅하여 균열의 끝점 찾기
    NMS로 분리된 3개 브랜치 선택
    
    Returns:
        lengths_px: 픽셀 단위 길이 리스트
        lengths_um: μm 단위 길이 리스트
        tips: 균열 끝점 좌표 리스트
        angles: 각 균열의 각도 리스트
    """
    cx, cy = center
    h, w = gray.shape
    
    # 엣지 검출
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 1.2), 50, 150)
    
    # 각도별 거리 측정
    angles = np.arange(0, 180, step_deg, dtype=np.float32)
    distances = np.zeros_like(angles)
    tips = [(int(cx), int(cy))] * len(angles)
    
    for i, angle in enumerate(angles):
        rad = np.deg2rad(angle)
        dx, dy = cos(rad), sin(rad)
        
        last_hit = None
        miss_count = 0
        
        # 레이 진행
        for r in range(5, int(max_radius_px)):
            x = int(round(cx + dx * r))
            y = int(round(cy + dy * r))
            
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            
            if edges[y, x] > 0:
                last_hit = (x, y)
                miss_count = 0
            else:
                miss_count += 1
                if miss_count > 4:  # 4픽셀 이상 엣지 없으면 중단
                    break
        
        if last_hit is not None:
            distances[i] = hypot(last_hit[0] - cx, last_hit[1] - cy)
            tips[i] = last_hit
    
    # NMS (Non-Maximum Suppression)로 3개 피크 선택
    picked_indices = []
    arr = distances.copy()
    window = max(1, int(sep_deg / step_deg))
    
    for _ in range(3):
        k = int(np.argmax(arr))
        if arr[k] <= 0:
            break
        
        picked_indices.append(k)
        
        # 선택된 피크 주변 억제
        left = max(0, k - window)
        right = min(len(arr), k + window + 1)
        arr[left:right] = 0
    
    # 결과 정리
    lengths_px = [float(distances[k]) for k in picked_indices]
    lengths_um = [d * um_per_px for d in lengths_px]
    tips_result = [tips[k] for k in picked_indices]
    angles_result = [float(angles[k]) for k in picked_indices]
    
    return lengths_px, lengths_um, tips_result, angles_result

# ==================== 메인 측정 함수 ====================
def measure_crack_length(image_path, 
                        scale_um=10.0,
                        bar_px=None,
                        um_per_px=None,
                        center=None,
                        max_um=None,
                        output_path=None,
                        visualize=True):
    """
    균열 길이 측정 메인 함수
    
    Args:
        image_path: 입력 이미지 경로 (PNG)
        scale_um: 스케일바의 실제 길이 (μm)
        bar_px: 스케일바 픽셀 길이 (None이면 자동 검출)
        um_per_px: 픽셀당 μm (직접 지정 시)
        center: 균열 중심 좌표 (x, y) 또는 None (자동 추정)
        max_um: 최대 측정 길이 (μm)
        output_path: 시각화 이미지 저장 경로
        visualize: 시각화 여부
    
    Returns:
        dict: 측정 결과
    """
    # 이미지 읽기
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 스케일 계산
    if um_per_px is not None:
        # 직접 지정
        scale_factor = um_per_px
        bar_bbox = None
    elif bar_px is not None:
        # 수동 스케일바 픽셀 지정
        if scale_um is None:
            raise ValueError("bar_px 사용 시 scale_um이 필요합니다")
        scale_factor = scale_um / bar_px
        bar_bbox = None
    else:
        # 자동 검출
        if scale_um is None:
            raise ValueError("자동 스케일바 검출 시 scale_um이 필요합니다")
        bar_width, bar_bbox = auto_detect_scale_bar(gray)
        scale_factor = scale_um / bar_width
    
    # 중심점 추정
    if center is None:
        center = estimate_crack_center(gray)
    
    # 최대 반경 계산
    if max_um is not None:
        max_radius_px = max_um / scale_factor
    else:
        max_radius_px = 120.0
    
    # 균열 길이 측정
    lengths_px, lengths_um, tips, angles = raycast_measure_cracks(
        gray, center, scale_factor, max_radius_px=max_radius_px
    )
    
    # 결과 딕셔너리
    result = {
        'image_path': image_path,
        'scale_um_per_px': scale_factor,
        'center': center,
        'cracks': [
            {
                'length_um': um,
                'length_px': px,
                'angle_deg': ang,
                'tip': tip
            }
            for um, px, ang, tip in zip(lengths_um, lengths_px, angles, tips)
        ],
        'bar_bbox': bar_bbox
    }
    
    # 시각화
    if visualize:
        vis = img.copy()
        
        # 중심점 표시
        cv2.circle(vis, (int(center[0]), int(center[1])), 6, (0, 255, 0), -1)
        
        # 스케일바 표시
        if bar_bbox is not None:
            x, y, w, h = bar_bbox
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(vis, f"{int(w)} px = {scale_um} um",
                       (x, max(15, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.55, (255, 0, 0), 1, cv2.LINE_AA)
        
        # 스케일 정보 표시
        scale_text = f"Scale: {scale_factor:.5f} um/px"
        cv2.putText(vis, scale_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        
        # 균열 표시
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255)]
        for i, crack in enumerate(result['cracks']):
            tip = crack['tip']
            color = colors[i % len(colors)]
            
            # 선 그리기
            cv2.line(vis, (int(center[0]), int(center[1])), 
                    (int(tip[0]), int(tip[1])), (0, 255, 0), 2, cv2.LINE_AA)
            
            # 라벨 표시
            label = f"{i+1}: {crack['length_um']:.3f} um @ {crack['angle_deg']:.0f}°"
            cv2.putText(vis, label, (int(tip[0])+5, int(tip[1])+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        
        # 저장
        if output_path is None:
            base, _ = os.path.splitext(image_path)
            output_path = base + "_result.png"
        
        imwrite_unicode(output_path, vis)
        result['output_path'] = output_path
    
    return result

# ==================== 배치 처리 ====================
def batch_process(dataset_dir, output_dir="outputs", scale_um=10.0):
    """
    데이터셋 전체를 배치 처리
    
    Args:
        dataset_dir: 데이터셋 디렉토리 (images_png, images_json 포함)
        output_dir: 출력 디렉토리
        scale_um: 스케일바의 실제 길이
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images_png"
    json_dir = dataset_path / "images_json"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = []
    
    # PNG 파일 처리
    for img_file in sorted(images_dir.glob("*.png")):
        print(f"\n처리 중: {img_file.name}")
        
        try:
            # 측정
            result = measure_crack_length(
                str(img_file),
                scale_um=scale_um,
                output_path=str(output_path / f"{img_file.stem}_result.png")
            )
            
            # JSON 파일 있으면 ground truth 비교
            json_file = json_dir / f"{img_file.stem}.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                
                gt_cracks = gt_data.get('cracks', [])
                measured = [c['length_um'] for c in result['cracks']]
                
                result['ground_truth'] = gt_cracks
                result['error'] = calculate_error(measured, gt_cracks)
                
                print(f"  측정값: {[f'{v:.3f}' for v in measured]} μm")
                print(f"  실제값: {[f'{v:.3f}' for v in gt_cracks]} μm")
                print(f"  오차: {result['error']:.2f}%")
            else:
                print(f"  측정값: {[f'{c['length_um']:.3f}' for c in result['cracks']]} μm")
                print(f"  (Ground truth 없음)")
            
            results.append(result)
            
        except Exception as e:
            print(f"  오류: {e}")
            continue
    
    # 전체 결과 저장
    summary_file = output_path / "measurement_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n총 {len(results)}개 이미지 처리 완료")
    print(f"결과 저장: {summary_file}")
    
    return results

def calculate_error(measured, ground_truth):
    """
    측정값과 실제값 사이의 평균 오차 계산 (%)
    """
    if not measured or not ground_truth:
        return None
    
    # 길이순 정렬
    measured = sorted(measured, reverse=True)
    gt = sorted(ground_truth, reverse=True)
    
    # 개수 맞추기
    min_len = min(len(measured), len(gt))
    measured = measured[:min_len]
    gt = gt[:min_len]
    
    # 상대 오차 계산
    errors = [abs(m - g) / g * 100 for m, g in zip(measured, gt)]
    return np.mean(errors)

# ==================== CLI ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='실리콘 웨이퍼 균열 길이 자동 측정')
    parser.add_argument('--image', type=str, help='입력 이미지 경로 (PNG)')
    parser.add_argument('--batch', type=str, help='배치 처리: 데이터셋 디렉토리 경로')
    parser.add_argument('--scale_um', type=float, default=10.0, 
                       help='스케일바의 실제 길이 (μm), 기본값=10.0')
    parser.add_argument('--bar_px', type=float, help='스케일바 픽셀 길이 (수동 지정)')
    parser.add_argument('--um_per_px', type=float, help='픽셀당 μm (직접 지정)')
    parser.add_argument('--center', type=str, help='균열 중심 좌표 "x,y"')
    parser.add_argument('--max_um', type=float, help='최대 측정 길이 (μm)')
    parser.add_argument('--output', type=str, help='출력 이미지 경로')
    parser.add_argument('--output_dir', type=str, default='outputs', 
                       help='배치 처리 출력 디렉토리')
    
    args = parser.parse_args()
    
    if args.batch:
        # 배치 처리
        batch_process(args.batch, args.output_dir, args.scale_um)
    
    elif args.image:
        # 단일 이미지 처리
        center = None
        if args.center:
            x, y = map(float, args.center.split(','))
            center = (x, y)
        
        result = measure_crack_length(
            image_path=args.image,
            scale_um=args.scale_um,
            bar_px=args.bar_px,
            um_per_px=args.um_per_px,
            center=center,
            max_um=args.max_um,
            output_path=args.output
        )
        
        print(f"\n=== 측정 결과 ===")
        print(f"이미지: {result['image_path']}")
        print(f"스케일: {result['scale_um_per_px']:.5f} μm/px")
        print(f"중심: ({result['center'][0]:.1f}, {result['center'][1]:.1f})")
        print(f"\n균열 길이:")
        for i, crack in enumerate(result['cracks'], 1):
            print(f"  {i}: {crack['length_um']:.3f} μm "
                  f"(각도: {crack['angle_deg']:.0f}°)")
        
        if 'output_path' in result:
            print(f"\n시각화 저장: {result['output_path']}")
    
    else:
        parser.print_help()
