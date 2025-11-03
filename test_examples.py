"""
균열 측정 시스템 사용 예제 및 테스트
"""

from crack_detector import measure_crack_length, batch_process
from pathlib import Path

def example_single_image():
    """단일 이미지 측정 예제"""
    print("=" * 60)
    print("단일 이미지 측정 예제")
    print("=" * 60)
    
    # 이미지 경로
    image_path = r"C:\Users\kimbr\Si_wafer\dataset\images_png\sem_02.png"
    
    # 측정 실행
    result = measure_crack_length(
        image_path=image_path,
        scale_um=10.0,  # 스케일바 실제 길이 10μm
        output_path=r"C:\Users\kimbr\Si_wafer\outputs\sem_02_result.png"
    )
    
    # 결과 출력
    print(f"\n이미지: {Path(result['image_path']).name}")
    print(f"스케일: {result['scale_um_per_px']:.5f} μm/px")
    print(f"중심: ({result['center'][0]:.1f}, {result['center'][1]:.1f})")
    print(f"\n균열 길이:")
    for i, crack in enumerate(result['cracks'], 1):
        print(f"  균열 {i}: {crack['length_um']:.3f} μm (각도: {crack['angle_deg']:.0f}°)")
    
    if 'output_path' in result:
        print(f"\n시각화 저장: {result['output_path']}")

def example_with_manual_params():
    """수동 파라미터 지정 예제"""
    print("\n" + "=" * 60)
    print("수동 파라미터 지정 예제")
    print("=" * 60)
    
    image_path = r"C:\Users\kimbr\Si_wafer\dataset\images_png\sem_03.png"
    
    result = measure_crack_length(
        image_path=image_path,
        scale_um=10.0,
        bar_px=87,  # 스케일바 픽셀 길이 수동 지정
        center=(620, 470),  # 중심 좌표 수동 지정
        max_um=12.0,  # 최대 측정 길이 제한
        output_path=r"C:\Users\kimbr\Si_wafer\outputs\sem_03_manual.png"
    )
    
    print(f"\n이미지: {Path(result['image_path']).name}")
    print(f"균열 길이: {[f'{c['length_um']:.3f}' for c in result['cracks']]} μm")

def example_batch_processing():
    """배치 처리 예제"""
    print("\n" + "=" * 60)
    print("배치 처리 예제")
    print("=" * 60)
    
    dataset_dir = r"C:\Users\kimbr\Si_wafer\dataset"
    output_dir = r"C:\Users\kimbr\Si_wafer\outputs"
    
    results = batch_process(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        scale_um=10.0
    )
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("측정 통계")
    print("=" * 60)
    
    errors = [r['error'] for r in results if 'error' in r and r['error'] is not None]
    if errors:
        print(f"평균 오차: {sum(errors)/len(errors):.2f}%")
        print(f"최소 오차: {min(errors):.2f}%")
        print(f"최대 오차: {max(errors):.2f}%")

def test_with_comparison():
    """Ground truth와 비교하며 테스트"""
    print("\n" + "=" * 60)
    print("Ground Truth 비교 테스트")
    print("=" * 60)
    
    import json
    
    # 테스트할 이미지들
    test_images = ['sem_02', 'sem_03', 'sem_04', 'sem_05']
    
    for img_name in test_images:
        print(f"\n[{img_name}]")
        
        image_path = f"C:/Users/kimbr/Si_wafer/dataset/images_png/{img_name}.png"
        json_path = f"C:/Users/kimbr/Si_wafer/dataset/images_json/{img_name}.json"
        
        try:
            # 측정
            result = measure_crack_length(
                image_path=image_path,
                scale_um=10.0,
                visualize=False  # 시각화 생략
            )
            
            measured = sorted([c['length_um'] for c in result['cracks']], reverse=True)
            
            # Ground truth 로드
            if Path(json_path).exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                gt_cracks = sorted(gt_data.get('cracks', []), reverse=True)
                
                print(f"  측정값: {[f'{v:.3f}' for v in measured]} μm")
                print(f"  실제값: {[f'{v:.3f}' for v in gt_cracks]} μm")
                
                # 오차 계산
                if measured and gt_cracks:
                    min_len = min(len(measured), len(gt_cracks))
                    errors = [abs(m - g) / g * 100 
                             for m, g in zip(measured[:min_len], gt_cracks[:min_len])]
                    avg_error = sum(errors) / len(errors)
                    print(f"  평균 오차: {avg_error:.2f}%")
            else:
                print(f"  측정값: {[f'{v:.3f}' for v in measured]} μm")
                print(f"  (Ground truth 없음)")
                
        except Exception as e:
            print(f"  오류: {e}")

if __name__ == "__main__":
    # 출력 디렉토리 생성
    Path("C:/Users/kimbr/Si_wafer/outputs").mkdir(exist_ok=True)
    
    # 예제 실행
    print("\n" + "=" * 60)
    print("균열 측정 시스템 테스트")
    print("=" * 60)
    
    # 1. 단일 이미지
    example_single_image()
    
    # 2. 수동 파라미터
    example_with_manual_params()
    
    # 3. Ground truth 비교
    test_with_comparison()
    
    # 4. 배치 처리 (시간이 오래 걸릴 수 있음)
    # example_batch_processing()
