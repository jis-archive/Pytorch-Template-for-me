from torch.utils.data import Dataset
import torch

class ReviewDataset(Dataset):
    """
    리뷰 텍스트 데이셋 클래스
    - BERT 모델 훈련/추론을 위한 PyTorch Dataset 구현
    - 텍스트 토크나이징 및 텐서 변환 처리
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        """
        데이터셋 초기화
        """
        self.texts, self.labels, self.tokenizer, self.max_length = (
            texts,
            labels,
            tokenizer,
            max_length,
        )

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        특정 인덱스의 데이터 아이템 반환
        """
        # 텍스트 토크나이징 및 패딩
        encoding = self.tokenizer(
            str(self.texts.iloc[idx]),
            truncation=True,  # 최대 길이 초과시 자르기
            padding="max_length",  # 최대 길이까지 패딩
            max_length=self.max_length,
            return_tensors="pt",  # PyTorch 텐서로 반환
        )

        # 기본 아이템 구성 (input_ids, attention_mask)
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # labels가 None이 아닌 경우에만 추가 (train/valid용)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

        return item