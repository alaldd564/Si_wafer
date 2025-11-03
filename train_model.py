"""
균열 길이 예측 모델 학습
Dataset의 이미지와 JSON을 사용하여 학습
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pickle

# OpenCV 기반 특징 추출
class CrackFeatureExtractor:
    """이미지에서 균열 특징 추출"""
    
    def __init__(self):
        self.scale_um = 10.0  # 기본 스케일바 길이
    
    def imread_unicode(self, path):
        """유니코드 경로 지원 이미지 읽기"""
        path = os.path.abspath(path)
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    def extract_features(self, image_path):
        """
        이미지에서 균열 관련 특징 추출
        
        Returns:
            features: 특징 벡터
        """
        img = self.imread_unicode(image_path)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. 스케일 추정
        scale_factor = self._estimate_scale(gray)
        
        # 2. 중심 추정
        center = self._estimate_center(gray)
        
        # 3. 방사형 특징 추출
        radial_features = self._extract_radial_features(gray, center)
        
        # 4. 엣지 밀도 특징
        edge_features = self._extract_edge_features(gray, center)
        
        # 5. 텍스처 특징
        texture_features = self._extract_texture_features(gray, center)
        
        # 특징 벡터 통합
        features = np.concatenate([
            [scale_factor],
            center / [w, h],  # 정규화된 중심
            radial_features,
            edge_features,
            texture_features
        ])
        
        return features
    
    def _estimate_scale(self, gray):
        """스케일바 검출 및 스케일 계산"""
        h, w = gray.shape
        y0 = int(h * 0.75)
        crop = gray[y0:, :]
        
        # 대비 향상
        crop_eq = cv2.equalizeHist(crop)
        _, th = cv2.threshold(crop_eq, 220, 255, cv2.THRESH_BINARY)
        
        # 형태학 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 2)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_width = w * 0.15  # 기본값
        best_score = -1
        
        for cnt in contours:
            x, y, wc, hc = cv2.boundingRect(cnt)
            aspect = wc / max(hc, 1)
            
            if aspect > 12 and wc > 0.10 * w:
                score = wc * aspect
                if score > best_score:
                    best_score = score
                    best_width = wc
        
        return self.scale_um / best_width
    
    def _estimate_center(self, gray):
        """균열 중심 추정"""
        h, w = gray.shape
        
        # 엣지 검출
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 1.2), 50, 150)
        
        # Hough 변환
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45, 25, 20)
        
        if lines is None or len(lines) < 2:
            return np.array([w/2, h/2])
        
        # 교차점 계산
        votes = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                pt = self._line_intersection(lines[i][0], lines[j][0])
                if pt is not None:
                    x, y = pt
                    if 0 <= x < w and 0 <= y < h:
                        votes.append([x, y])
        
        if not votes:
            return np.array([w/2, h/2])
        
        # 클러스터링
        votes = np.array(votes)
        return np.median(votes, axis=0)
    
    def _line_intersection(self, line1, line2):
        """두 선분의 교차점"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return None
        
        x = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        y = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
        
        return (x, y)
    
    def _extract_radial_features(self, gray, center):
        """방사형 특징 추출 (360도)"""
        cx, cy = center
        h, w = gray.shape
        
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 1.2), 50, 150)
        
        angles = np.arange(0, 180, 2)  # 90개 방향
        distances = []
        
        for angle in angles:
            rad = np.deg2rad(angle)
            dx, dy = np.cos(rad), np.sin(rad)
            
            max_dist = 0
            for r in range(5, 120):
                x = int(cx + dx * r)
                y = int(cy + dy * r)
                
                if x < 0 or x >= w or y < 0 or y >= h:
                    break
                
                if edges[y, x] > 0:
                    max_dist = r
            
            distances.append(max_dist)
        
        distances = np.array(distances)
        
        # 통계적 특징
        features = [
            np.mean(distances),
            np.std(distances),
            np.max(distances),
            np.percentile(distances, 75),
            np.percentile(distances, 25),
        ]
        
        # 상위 3개 피크
        sorted_dist = np.sort(distances)[-3:]
        features.extend(sorted_dist)
        
        return np.array(features)
    
    def _extract_edge_features(self, gray, center):
        """엣지 밀도 특징"""
        cx, cy = int(center[0]), int(center[1])
        
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 1.2), 50, 150)
        
        # 중심 주변 영역별 엣지 밀도
        features = []
        for radius in [30, 60, 90, 120]:
            mask = np.zeros_like(edges)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            if radius > 30:
                cv2.circle(mask, (cx, cy), radius-30, 0, -1)
            
            edge_density = np.sum(edges[mask > 0]) / np.sum(mask > 0)
            features.append(edge_density)
        
        return np.array(features)
    
    def _extract_texture_features(self, gray, center):
        """텍스처 특징"""
        cx, cy = int(center[0]), int(center[1])
        h, w = gray.shape
        
        # 중심 주변 영역
        size = 100
        x1 = max(0, cx - size)
        x2 = min(w, cx + size)
        y1 = max(0, cy - size)
        y2 = min(h, cy + size)
        
        roi = gray[y1:y2, x1:x2]
        
        # 통계적 특징
        features = [
            np.mean(roi),
            np.std(roi),
            np.percentile(roi, 25),
            np.percentile(roi, 75),
        ]
        
        return np.array(features)


class CrackLengthPredictor:
    """균열 길이 예측 모델"""
    
    def __init__(self):
        self.feature_extractor = CrackFeatureExtractor()
        self.feature_means = None
        self.feature_stds = None
        self.weights = None
        self.bias = None
    
    def train(self, dataset_dir, verbose=True):
        """
        데이터셋으로 모델 학습
        
        Args:
            dataset_dir: 데이터셋 디렉토리 (images_png, images_json 포함)
            verbose: 진행상황 출력 여부
        """
        dataset_path = Path(dataset_dir)
        images_dir = dataset_path / "images_png"
        json_dir = dataset_path / "images_json"
        
        if not images_dir.exists():
            raise ValueError(f"이미지 디렉토리가 없습니다: {images_dir}")
        
        # 데이터 수집
        X = []  # 특징
        y = []  # 균열 길이들
        
        image_files = sorted(images_dir.glob("*.png"))
        
        if verbose:
            print(f"학습 데이터 로드 중... (총 {len(image_files)}개)")
            image_files = tqdm(image_files)
        
        for img_file in image_files:
            json_file = json_dir / f"{img_file.stem}.json"
            
            if not json_file.exists():
                continue
            
            try:
                # 특징 추출
                features = self.feature_extractor.extract_features(str(img_file))
                
                # Ground truth 로드
                with open(json_file, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                
                crack_lengths = gt_data.get('cracks', [])
                
                if len(crack_lengths) != 3:
                    continue
                
                X.append(features)
                y.append(crack_lengths)
                
            except Exception as e:
                if verbose:
                    print(f"  오류 ({img_file.name}): {e}")
                continue
        
        if len(X) == 0:
            raise ValueError("학습 데이터가 없습니다.")
        
        X = np.array(X)
        y = np.array(y)
        
        if verbose:
            print(f"\n학습 데이터: {len(X)}개")
            print(f"특징 차원: {X.shape[1]}")
        
        # 특징 정규화
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self.feature_means) / self.feature_stds
        
        # 선형 회귀로 학습 (각 균열 길이별)
        self.weights = []
        self.bias = []
        
        for i in range(3):  # 3개 균열
            y_i = y[:, i]
            
            # 최소제곱법
            X_with_bias = np.column_stack([X_normalized, np.ones(len(X_normalized))])
            params = np.linalg.lstsq(X_with_bias, y_i, rcond=None)[0]
            
            self.weights.append(params[:-1])
            self.bias.append(params[-1])
        
        self.weights = np.array(self.weights)
        self.bias = np.array(self.bias)
        
        # 학습 정확도 평가
        y_pred = self.predict_features(X)
        train_error = self._calculate_error(y_pred, y)
        
        if verbose:
            print(f"\n학습 완료!")
            print(f"평균 오차: {train_error:.2f}%")
        
        return train_error
    
    def predict(self, image_path):
        """
        이미지로부터 균열 길이 예측
        
        Args:
            image_path: 이미지 경로
        
        Returns:
            lengths: 3개 균열의 길이 (μm)
        """
        if self.weights is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 실행하세요.")
        
        # 특징 추출
        features = self.feature_extractor.extract_features(image_path)
        
        # 예측
        lengths = self.predict_features(features[np.newaxis, :])[0]
        
        return lengths
    
    def predict_features(self, features):
        """특징 벡터로부터 예측"""
        # 정규화
        X = (features - self.feature_means) / self.feature_stds
        
        # 예측
        predictions = []
        for i in range(3):
            pred = np.dot(X, self.weights[i]) + self.bias[i]
            predictions.append(pred)
        
        predictions = np.array(predictions).T
        return predictions
    
    def _calculate_error(self, predicted, actual):
        """평균 상대 오차 계산 (%)"""
        errors = np.abs(predicted - actual) / actual * 100
        return np.mean(errors)
    
    def save_model(self, filepath):
        """모델 저장"""
        model_data = {
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'weights': self.weights,
            'bias': self.bias,
            'scale_um': self.feature_extractor.scale_um
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"모델 저장: {filepath}")
    
    def load_model(self, filepath):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.feature_means = model_data['feature_means']
        self.feature_stds = model_data['feature_stds']
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.feature_extractor.scale_um = model_data['scale_um']
        
        print(f"모델 로드: {filepath}")


def main():
    """학습 실행"""
    print("=" * 60)
    print("균열 길이 예측 모델 학습")
    print("=" * 60)
    
    # 데이터셋 경로
    dataset_dir = r"C:\Users\kimbr\Si_wafer\dataset"
    model_path = r"C:\Users\kimbr\Si_wafer\crack_model.pkl"
    
    # 모델 생성 및 학습
    predictor = CrackLengthPredictor()
    
    print("\n[1단계] 학습 시작...")
    train_error = predictor.train(dataset_dir, verbose=True)
    
    print("\n[2단계] 모델 저장...")
    predictor.save_model(model_path)
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"모델 파일: {model_path}")
    print(f"학습 오차: {train_error:.2f}%")
    print("=" * 60)
    
    print("\n이제 'python predict.py --image test.tif' 로 예측할 수 있습니다.")


if __name__ == "__main__":
    main()
