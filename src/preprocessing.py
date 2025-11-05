import re
import pandas as pd
from sklearn.model_selection import train_test_split

# 텍스트 전처리 파이프라인 클래스 구성
class TextPreprocessingPipeline:
    """
    텍스트 전처리 파이프라인 클래스
    - 기본 전처리와 학습 데이터 기반 고급 전처리를 통합 관리
    - 재사용 가능하고 확장 가능한 구조
    """

    def __init__(self):
        self.is_fitted = False
        self.vocab_info = {}
        self.label_patterns = {}

    def basic_preprocess(self, texts):
        """기본 전처리 (clean_text + normalize 기능)"""
        processed_texts = []
        for text in texts:
            # 기본 텍스트 정리
            cleaned = self._clean_text(text)
            # 정규화
            normalized = self._normalize_text(cleaned)
            processed_texts.append(normalized)
        return processed_texts

    def _clean_text(self, text):
        """기존 clean_text 함수 내용"""
        if pd.isna(text):
            return ""

        text = str(text).strip()
        # text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", "", text)
        text = re.sub(r"([ㅋㅎ])\1{2,}", r"\1\1", text)
        text = re.sub(r"([ㅠㅜㅡ])\1{2,}", r"\1\1", text)
        text = re.sub(r"(ㅋ{2,}|ㅎ{2,}|ㅠ{2,}|ㅜ{2,}|ㅉ{2,}|ㅡ{2,})|[ㄱ-ㅎㅏ-ㅣ]+", lambda m: m.group(1) or "", text)
        text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
        text = re.sub(r"[^\w\s가-힣.,!?ㅋㅎㅠㅜㅡ~\-]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _normalize_text(self, text):
        """텍스트 정규화 함수"""
        # 소문자 변환
        text = text.lower()

        # 구두점 정규화
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[,]{2,}", ",", text)
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"([.,!?])\s+", r"\1 ", text)

        # 특수문자 정리
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def fit(self, texts, labels=None):
        """학습 데이터로부터 전처리 정보 학습"""
        print("학습 데이터 기반 전처리 정보 수집 중...")

        # NOTE: 학습 데이터를 분석하여 전처리에 필요한 정보를 수집하고 저장하는 코드를 직접 구현해서 추가해보세요!
        # 예시:
        # - 도메인 특화 패턴 분석 (영화 리뷰 특성)
        # - 빈도 기반 노이즈 패턴 식별
        # - 라벨별 텍스트 특성 분석
        # - 어휘 사전 구축
        # - 정규화 규칙 최적화

        self.is_fitted = True
        print("✓ 전처리 파이프라인 학습 완료")

    def transform(self, texts):
        """전처리 적용"""
        if not self.is_fitted:
            print(
                "Warning: 파이프라인이 학습되지 않았습니다. 기본 전처리만 적용합니다."
            )

        return self.basic_preprocess(texts)

    def fit_transform(self, texts, labels=None):
        """학습과 변환을 동시에 수행"""
        self.fit(texts, labels)
        return self.transform(texts)

def load_and_preprocess(data_path, random_state=42):
    # 전처리 파이프라인 인스턴스 생성
    preprocessor = TextPreprocessingPipeline()
    print("텍스트 전처리 파이프라인 클래스 구성 완료")
    print("현재: 기본 전처리 기능 구현됨")

    df = pd.read_csv(data_path)

    # 데이터 분할 - 텍스트 전처리 파이프라인 적용
    X = df["review"]  # 원본 텍스트 데이터 사용 (파이프라인에서 전처리 수행)
    y = df["label"]
    ids = df["ID"]

    # 훈련/검증 데이터 분할 (train 80%, val 20%) - 계층 분할로 클래스 분포 유지
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X, y, ids, test_size=0.2, random_state=random_state, stratify=y
    )

    # 텍스트 전처리 파이프라인 적용
    print("훈련 데이터에 대한 전처리 파이프라인 학습 및 적용...")
    X_train_processed = preprocessor.fit_transform(X_train.tolist(), y_train.tolist())

    print("검증 데이터에 전처리 파이프라인 적용...")
    X_val_processed = preprocessor.transform(X_val.tolist())

    # 원본 데이터프레임 구조로 분할된 데이터 생성 - 모델 학습용 형태
    train_data = pd.DataFrame(
        {"ID": ids_train, "review": X_train_processed, "label": y_train}
    ).reset_index(drop=True)

    val_data = pd.DataFrame(
        {"ID": ids_val, "review": X_val_processed, "label": y_val}
    ).reset_index(drop=True)

    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")

    # 계층 분할이 올바르게 수행되었는지 검증 - 클래스 분포 확인
    print("\n클래스 분포 검증:")
    print("원본 데이터:")
    original_distribution = y.value_counts(normalize=True).sort_index()
    original_counts = y.value_counts().sort_index()
    for idx, val in original_distribution.items():
        count = original_counts[idx]
        print(f"  클래스 {idx}: {count}개 ({val * 100:.1f}%)")

    print("\n훈련 데이터:")
    train_distribution = train_data["label"].value_counts(normalize=True).sort_index()
    train_counts = train_data["label"].value_counts().sort_index()
    for idx, val in train_distribution.items():
        count = train_counts[idx]
        print(f"  클래스 {idx}: {count}개 ({val * 100:.1f}%)")

    print("\n검증 데이터:")
    val_distribution = val_data["label"].value_counts(normalize=True).sort_index()
    val_counts = val_data["label"].value_counts().sort_index()
    for idx, val in val_distribution.items():
        count = val_counts[idx]
        print(f"  클래스 {idx}: {count}개 ({val * 100:.1f}%)")

    # 전처리 결과 확인
    print("\n전처리 결과 샘플:")
    for i in range(3):
        print(f"원본: {X_train.iloc[i]}")
        print(f"전처리: {X_train_processed[i]}")
        print()             
    
    return train_data, val_data

def load_and_preprocess_test(test_texts, random_state=42):
    # 전처리 파이프라인 인스턴스 생성
    preprocessor = TextPreprocessingPipeline()
    print("텍스트 전처리 파이프라인 클래스 구성 완료")
    print("현재: 기본 전처리 기능 구현됨")

    return preprocessor.transform(test_texts)

## train, val, test 모두 적용 가능하게 파편화 하자.