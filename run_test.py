"""
간단한 실행 테스트
모든 기능을 한번에 테스트합니다
"""

from pathlib import Path
import sys

print("=" * 60)
print("균열 측정 시스템 통합 테스트")
print("=" * 60)

# 1. 환경 체크
print("\n[1단계] 환경 체크")
print("-" * 60)

try:
    import cv2
    print("✓ OpenCV")
except:
    print("✗ OpenCV - pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy")
except:
    print("✗ NumPy - pip install numpy")
    sys.exit(1)

try:
    from tqdm import tqdm
    print("✓ tqdm")
except:
    print("✗ tqdm - pip install tqdm")

try:
    import tifffile
    print("✓ tifffile")
except:
    print("✗ tifffile - pip install tifffile")

# 2. 파일 체크
print("\n[2단계] 파일 체크")
print("-" * 60)

dataset_dir = Path(r"C:\Users\kimbr\Si_wafer\dataset")
images_png = dataset_dir / "images_png"
images_json = dataset_dir / "images_json"

if images_png.exists():
    png_count = len(list(images_png.glob("*.png")))
    print(f"✓ PNG 이미지: {png_count}개")
else:
    print("✗ images_png 폴더 없음")

if images_json.exists():
    json_count = len(list(images_json.glob("*.json")))
    print(f"✓ JSON 파일: {json_count}개")
else:
    print("✗ images_json 폴더 없음")

# 3. 모델 체크
print("\n[3단계] 모델 체크")
print("-" * 60)

model_path = Path(r"C:\Users\kimbr\Si_wafer\crack_model.pkl")

if model_path.exists():
    print(f"✓ 학습된 모델 존재: {model_path.name}")
    print("\n바로 예측 가능:")
    print("  python predict.py --image test.png")
else:
    print("✗ 학습된 모델 없음")
    print("\n먼저 학습 필요:")
    print("  python train_model.py")

# 4. 테스트 이미지 체크
print("\n[4단계] 테스트 이미지 체크")
print("-" * 60)

root = Path(r"C:\Users\kimbr\Si_wafer")
test_tif = root / "test.tif"
test_png = root / "test.png"

if test_tif.exists():
    print(f"✓ test.tif 존재")
    print("  PNG 변환: python convert_tif.py --input test.tif --output test.png")
elif test_png.exists():
    print(f"✓ test.png 존재")
    print("  바로 예측 가능")
else:
    print("✗ 테스트 이미지 없음 (test.tif 또는 test.png)")
    print("  대신 dataset 이미지 사용 가능:")
    
    if png_count > 0:
        first_image = sorted(images_png.glob("*.png"))[0]
        print(f"  python predict.py --image {first_image} --accuracy")

# 5. 추천 실행 순서
print("\n" + "=" * 60)
print("추천 실행 순서")
print("=" * 60)

if not model_path.exists():
    print("\n1️⃣ 모델 학습 (최초 1회)")
    print("   python train_model.py")
    print()

if test_tif.exists() and not test_png.exists():
    print("2️⃣ TIF → PNG 변환")
    print("   python convert_tif.py --input test.tif --output test.png")
    print()

print("3️⃣ 예측 실행")
if test_png.exists() or test_tif.exists():
    print("   python predict.py --image test.png")
elif png_count > 0:
    first_image = sorted(images_png.glob("*.png"))[0]
    print(f"   python predict.py --image {first_image} --accuracy")
print()

print("4️⃣ 배치 예측 (전체 평가)")
print("   python predict.py --batch dataset")

print("\n" + "=" * 60)
