"""
TIF를 PNG로 변환하는 유틸리티
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_tif_to_png(tif_path, png_path=None):
    """
    TIF 파일을 PNG로 변환
    
    Args:
        tif_path: TIF 파일 경로
        png_path: PNG 저장 경로 (None이면 자동 생성)
    """
    # TIF 읽기
    try:
        import tifffile
        img = tifffile.imread(tif_path)
    except:
        # Fallback: OpenCV로 읽기
        img_data = np.fromfile(tif_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"TIF 파일을 읽을 수 없습니다: {tif_path}")
    
    # 8bit로 변환 (필요시)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    
    # PNG 경로
    if png_path is None:
        tif_path = Path(tif_path)
        png_path = tif_path.parent / f"{tif_path.stem}.png"
    
    # PNG 저장
    if isinstance(png_path, Path):
        png_path = str(png_path)
    
    # 유니코드 경로 지원
    success, encoded = cv2.imencode('.png', img)
    if success:
        encoded.tofile(png_path)
        return True
    return False


def batch_convert_tif_to_png(input_dir, output_dir):
    """
    디렉토리의 모든 TIF를 PNG로 변환
    
    Args:
        input_dir: TIF 파일들이 있는 디렉토리
        output_dir: PNG 저장 디렉토리
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    tif_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    
    print(f"TIF → PNG 변환 중... (총 {len(tif_files)}개)")
    
    for tif_file in tqdm(tif_files):
        png_file = output_path / f"{tif_file.stem}.png"
        
        try:
            convert_tif_to_png(str(tif_file), str(png_file))
        except Exception as e:
            print(f"  오류 ({tif_file.name}): {e}")
    
    print(f"\n변환 완료: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TIF를 PNG로 변환')
    parser.add_argument('--input', type=str, required=True, help='TIF 파일 또는 디렉토리')
    parser.add_argument('--output', type=str, help='PNG 저장 경로 또는 디렉토리')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 단일 파일
        output_path = args.output if args.output else None
        convert_tif_to_png(str(input_path), output_path)
        print(f"변환 완료: {output_path if output_path else input_path.with_suffix('.png')}")
    
    elif input_path.is_dir():
        # 디렉토리
        output_dir = args.output if args.output else str(input_path.parent / f"{input_path.name}_png")
        batch_convert_tif_to_png(str(input_path), output_dir)
    
    else:
        print(f"오류: 경로를 찾을 수 없습니다: {input_path}")
