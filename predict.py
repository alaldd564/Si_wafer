"""
학습된 모델로 균열 길이 예측
루트의 이미지 파일로 예측 실행
"""

import argparse
from pathlib import Path
from train_model import CrackLengthPredictor


def predict_image(image_path, model_path, show_accuracy=False):
    """
    이미지에서 균열 길이 예측
    
    Args:
        image_path: 예측할 이미지 경로
        model_path: 학습된 모델 경로
        show_accuracy: Ground truth가 있으면 정확도 표시
    """
    # 모델 로드
    predictor = CrackLengthPredictor()
    predictor.load_model(model_path)
    
    print("=" * 60)
    print(f"균열 길이 예측: {Path(image_path).name}")
    print("=" * 60)
    
    # 예측
    print("\n예측 중...")
    lengths = predictor.predict(image_path)
    
    # 결과 출력
    print("\n[예측 결과]")
    for i, length in enumerate(lengths, 1):
        print(f"  브랜치 {i}: {length:.3f} μm")
    
    print(f"\n평균 길이: {lengths.mean():.3f} μm")
    
    # Ground truth와 비교 (있으면)
    if show_accuracy:
        import json
        
        # JSON 파일 경로 추정
        image_name = Path(image_path).stem
        json_path = Path(image_path).parent.parent / "dataset" / "images_json" / f"{image_name}.json"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            gt_lengths = gt_data.get('cracks', [])
            
            if len(gt_lengths) == 3:
                print("\n" + "-" * 60)
                print("[실제값 비교]")
                
                gt_sorted = sorted(gt_lengths, reverse=True)
                pred_sorted = sorted(lengths, reverse=True)
                
                errors = []
                for i, (pred, actual) in enumerate(zip(pred_sorted, gt_sorted), 1):
                    error = abs(pred - actual) / actual * 100
                    errors.append(error)
                    print(f"  브랜치 {i}: 예측 {pred:.3f} μm | 실제 {actual:.3f} μm | 오차 {error:.2f}%")
                
                avg_error = sum(errors) / len(errors)
                print(f"\n평균 오차: {avg_error:.2f}%")
                
                # 정확도 (100 - 오차)
                accuracy = 100 - avg_error
                print(f"정확도: {accuracy:.2f}%")
    
    print("=" * 60)


def batch_predict(dataset_dir, model_path):
    """
    여러 이미지 배치 예측
    
    Args:
        dataset_dir: 데이터셋 디렉토리
        model_path: 학습된 모델 경로
    """
    predictor = CrackLengthPredictor()
    predictor.load_model(model_path)
    
    images_dir = Path(dataset_dir) / "images_png"
    json_dir = Path(dataset_dir) / "images_json"
    
    print("=" * 60)
    print("배치 예측")
    print("=" * 60)
    
    image_files = sorted(images_dir.glob("*.png"))
    
    total_error = 0
    count = 0
    
    for img_file in image_files:
        json_file = json_dir / f"{img_file.stem}.json"
        
        if not json_file.exists():
            continue
        
        # 예측
        try:
            lengths = predictor.predict(str(img_file))
            
            # Ground truth 로드
            import json
            with open(json_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            gt_lengths = gt_data.get('cracks', [])
            
            if len(gt_lengths) != 3:
                continue
            
            # 오차 계산
            gt_sorted = sorted(gt_lengths, reverse=True)
            pred_sorted = sorted(lengths, reverse=True)
            
            errors = [abs(p - g) / g * 100 for p, g in zip(pred_sorted, gt_sorted)]
            avg_error = sum(errors) / len(errors)
            
            print(f"\n{img_file.name}")
            print(f"  예측: {[f'{l:.3f}' for l in pred_sorted]} μm")
            print(f"  실제: {[f'{l:.3f}' for l in gt_sorted]} μm")
            print(f"  오차: {avg_error:.2f}%")
            
            total_error += avg_error
            count += 1
            
        except Exception as e:
            print(f"\n{img_file.name}: 오류 - {e}")
            continue
    
    if count > 0:
        print("\n" + "=" * 60)
        print(f"전체 평균 오차: {total_error/count:.2f}%")
        print(f"전체 평균 정확도: {100 - total_error/count:.2f}%")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='학습된 모델로 균열 길이 예측')
    parser.add_argument('--image', type=str, help='예측할 이미지 경로')
    parser.add_argument('--model', type=str, default='crack_model.pkl', 
                       help='모델 파일 경로 (기본값: crack_model.pkl)')
    parser.add_argument('--accuracy', action='store_true', 
                       help='Ground truth가 있으면 정확도 표시')
    parser.add_argument('--batch', type=str, 
                       help='배치 예측: 데이터셋 디렉토리')
    
    args = parser.parse_args()
    
    # 모델 경로
    model_path = args.model
    if not Path(model_path).exists():
        print(f"오류: 모델 파일이 없습니다: {model_path}")
        print("먼저 'python train_model.py'로 모델을 학습하세요.")
        return
    
    if args.batch:
        # 배치 예측
        batch_predict(args.batch, model_path)
    
    elif args.image:
        # 단일 이미지 예측
        if not Path(args.image).exists():
            print(f"오류: 이미지 파일이 없습니다: {args.image}")
            return
        
        predict_image(args.image, model_path, args.accuracy)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
