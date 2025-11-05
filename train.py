import argparse
import collections
import torch
import numpy as np
import os
import pandas as pd

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
    Trainer,
    TrainingArguments,
    AddedToken
)

import src.metrics as module_metric
from src.dataset import ReviewDataset
from src.config import ConfigParser
from src.preprocessing import load_and_preprocess, load_and_preprocess_test

def train_model(config, ensemble_model_path=None):
    logger = config.get_logger('train')

    model_path = ensemble_model_path if ensemble_model_path else config['arch']['args']['model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=config['arch']['args'].get('num_labels', 2)    
    )

    new_domain_tokens =[
        "ㅠㅠ", "ㅜㅜ",
        "ㅉㅉ", "ㅡㅡ", "-_-",
    ]

    added = [AddedToken(t, single_word=True, lstrip=False, rstrip=False, normalized=False) for t in new_domain_tokens]
    num_added = tokenizer.add_tokens(added)
    print(f"Added {num_added} new tokens to vocab. (total size: {len(tokenizer)})")
                           
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"✅ Model embedding resized to {len(tokenizer)}") 


    compute_metrics = getattr(module_metric, 'compute_metrics')
    logger.info(model)

    train_data, val_data = load_and_preprocess('./data/train.csv')

    train_dataset = ReviewDataset(train_data["review"], train_data["label"], tokenizer, config['data_loader']['args']['max_length'])
    eval_dataset = ReviewDataset(val_data["review"], val_data["label"], tokenizer, config['data_loader']['args']['max_length'])

    training_args = TrainingArguments(
        output_dir=config['trainer']['save_dir'],
        num_train_epochs=config['trainer']['epochs'],

        per_device_train_batch_size=config['data_loader']['args']['train_batch_size'],
        per_device_eval_batch_size=config['data_loader']['args']['eval_batch_size'],

        warmup_steps=config['trainer']['args'].get('warmup_steps', 0),
        weight_decay=config['trainer']['args'].get('weight_decay', 0.0),
        learning_rate=config['optimizer']['args'].get('lr', 5e-5),

        logging_strategy=config['trainer']['args'].get('logging_strategy', 'epoch'),
        logging_steps=config['trainer']['args'].get('logging_steps', 100),
        eval_strategy=config['trainer']['args'].get('eval_strategy', 'epoch'),

        save_strategy="epoch" if config['trainer'].get('is_save_model', True) else "no",
        load_best_model_at_end=config['trainer'].get('is_save_model', True),
        metric_for_best_model="accuracy" if config['trainer'].get('is_save_model', True) else None,
        greater_is_better=True,
        save_total_limit=2 if config['trainer']['is_save_model'] else 0,

        report_to="wandb" if config['trainer']['args'].get('use_wandb', False) else "none",
        run_name=config['trainer']['args'].get('run_name', "hf-train-run"),

        seed=config['trainer']['args'].get('random_state', 42),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=config['data_loader']['args'].get('num_workers', 2),
        remove_unused_columns=config['trainer']['args'].get('remove_unused_columns', False),
        push_to_hub=config['trainer']['args'].get('push_to_hub', False),
        gradient_accumulation_steps=config['trainer']['args'].get('gradient_accumulation_steps', 1),
        logging_first_step=config['trainer']['args'].get('logging_first_step', True)
    )

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 훈련 정보 출력
    print(f"훈련 샘플: {len(train_dataset):,}개")
    print(f"검증 샘플: {len(eval_dataset):,}개")
    print(f"훈련 에포크: {training_args.num_train_epochs}회")
    print(f"배치 크기: {config['data_loader']['args']['train_batch_size']} (훈련) / {config['data_loader']['args']['eval_batch_size']} (검증)")
    print(f"학습률: {config['optimizer']['args'].get('lr', 5e-5)}")
    print(f"시드값: {config['trainer']['args'].get('random_state', 42)}")

    print(f"wandb 사용: {config['trainer']['args'].get('use_wandb', False)}")

    # 훈련 실행
    try:
        training_results = trainer.train()
        print("\n훈련 완료")
        print(f"최종 훈련 손실: {training_results.training_loss:.4f}")
        eval_results = trainer.evaluate()
    
        # 결과 추출
        accuracy = eval_results.get('eval_accuracy', 0.0)

        # 훈련 로그 정보 출력
        if hasattr(training_results, "log_history"):
            print(f"총 훈련 스텝: {training_results.global_step}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 훈련이 중단되었습니다.")
        raise
    except Exception as e:
        print(f"\n훈련 중 오류 발생: {str(e)}")
        raise

    return model, tokenizer, accuracy
    

def ensemble_train(config):
    logger = config.get_logger('ensemble')
    ensemble_paths = config['ensemble']['model_paths']

    trained_models = []
    tokenizers = []
    model_accuracies = []
    for path in ensemble_paths:
        tmp_model, tmp_tokenizer, tmp_accuracy = train_model(config, ensemble_model_path=path)

        trained_models.append(tmp_model)
        tokenizers.append(tmp_tokenizer)
        model_accuracies.append(tmp_accuracy)

        torch.cuda.empty_cache()
    return trained_models, tokenizers, model_accuracies

def setup_device(args):
    """
    GPU 설정 및 정보 출력
    """
    # 1️⃣ 선택된 GPU 인덱스 환경변수에 반영
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print(f"✅ 지정된 GPU: {args.device}")
    else:
        print("⚠️ GPU 인덱스가 지정되지 않았습니다. 모든 GPU 또는 CPU를 사용합니다.")

    # 2️⃣ PyTorch 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3️⃣ GPU 상태 점검
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU {gpu_count}개 사용 가능: {device}")
        for i in range(gpu_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ CUDA 사용 불가 - CPU로 훈련 진행")

    return device


def inference_ensemble(config, trained_models, tokenizers, model_accuracies):
    # ========================================
    # 🤖 앙상블 모델 Setup (Weighted Soft Voting - Exponential)
    # ========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LABEL_MAPPING = {0: "강한 부정", 1: "약한 부정", 2: "약한 긍정", 3: "강한 긍정"}
    model_names = [
        "klue/roberta-base", 
        "klue/bert-base", 
        "kykim/bert-kor-base", 
        "beomi/kcbert-base", 
        "monologg/koelectra-base-v3-discriminator",
    ]   

    print(model_accuracies)
    df_test = pd.read_csv("./data/test.csv")

    print(f"\n{'='*60}")
    print(f"🎯 앙상블 추론 준비 (Weighted Soft Voting - Exponential)")
    print(f"{'='*60}")
    print(f"앙상블 모델 수: {len(trained_models)}개")
        

   
    # Exponential 방법으로 가중치 계산

    print(f"\n{'='*60}")
    print("🎯 Exponential 방법으로 가중치 계산")
    print(f"{'='*60}")

    model_accuracies = np.array(model_accuracies)

    # Exponential 가중치 계산
    EXP_SCALE = 30  # 스케일 파라미터
    model_weights = np.exp(model_accuracies * EXP_SCALE) / np.exp(model_accuracies * EXP_SCALE).sum()

    print(f"\n⚙️ Exponential Scale: {EXP_SCALE}")
    print(f"\n각 모델의 가중치:")
    print("-" * 80)
    for idx, (name, acc, weight) in enumerate(zip(model_names, model_accuracies, model_weights)):
        model_name = name.split('/')[-1]
        print(f"  [{idx+1}] {model_name:30s} | Accuracy: {acc:.4f} | Weight: {weight:.4f} ({weight*100:.1f}%)")
    print("-" * 80)

    # 가중치 통계
    print(f"\n📊 가중치 통계:")
    print(f"   합계: {model_weights.sum():.4f}")
    print(f"   최대: {model_weights.max():.4f} ({model_weights.max()*100:.1f}%)")
    print(f"   최소: {model_weights.min():.4f} ({model_weights.min()*100:.1f}%)")
    print(f"   최대/최소 비율: {model_weights.max() / model_weights.min():.2f}배")
    print(f"   표준편차: {model_weights.std():.4f}")


    # 테스트 데이터 준비

    print(f"\n{'='*60}")
    print("📋 테스트 데이터 준비")
    print(f"{'='*60}")

    # 테스트 데이터 전처리 파이프라인 적용
    print("\n테스트 데이터에 전처리 파이프라인 적용...")
    test_texts = df_test["review"].tolist()
    test_processed = load_and_preprocess_test(test_texts)

    # 전처리된 테스트 데이터 준비
    test_data = pd.DataFrame(
        {
            "ID": df_test["ID"],
            "review": test_processed,
            "label": [-1] * len(df_test),  # 테스트 데이터는 레이블 없음 (더미 값)
        }
    ).reset_index(drop=True)

    print(f"테스트 데이터: {len(test_data):,} 샘플")

    
    # 가중 앙상블 추론 수행 (Exponential 가중치)

    print(f"\n{'='*60}")
    print("🔮 가중 앙상블 추론 시작 (Exponential 가중치 적용)...")
    print(f"{'='*60}")

    all_model_probs = []  # 각 모델의 가중 확률값 저장

    for idx, (model, tokenizer, weight) in enumerate(zip(trained_models, tokenizers, model_weights)):
        model_name = model_names[idx].split('/')[-1]
        print(f"\n[{idx+1}/{len(trained_models)}] 🤖 {model_name} 예측 중... (weight: {weight:.4f})")
        
        # 모델을 GPU로 이동 및 eval 모드
        model.eval()
        model = model.to(device)
        
        # 해당 모델의 토크나이저로 데이터셋 생성
        test_dataset = ReviewDataset(
            test_data["review"], None, tokenizer, config['data_loader']['args']['max_length']
        )
        
        # DataLoader 생성
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['data_loader']['args']['eval_batch_size'],
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
        )
        
        # 배치별 예측 수행
        model_probs = []
        with torch.no_grad():
            for batch in test_dataloader:
                # 배치를 디바이스로 이동
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 예측
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)  # 확률로 변환
                
                # *** Exponential 가중치 적용 ***
                weighted_probs = probs.cpu().numpy() * weight
                model_probs.append(weighted_probs)
        
        # 모든 배치 결과 합치기
        model_probs = np.vstack(model_probs)
        all_model_probs.append(model_probs)
        
        print(f"   ✅ {model_name} 예측 완료 (shape: {model_probs.shape})")
        
        # GPU 메모리 해제
        model = model.cpu()
        torch.cuda.empty_cache()

    # ========================================
    # 🎯 Weighted Soft Voting: 가중치 적용된 확률 합산
    # ========================================

    print(f"\n{'='*60}")
    print("📊 Weighted Soft Voting 계산 중...")
    print(f"{'='*60}")

    # 각 모델의 가중치가 적용된 확률을 합산
    ensemble_probs = np.sum(all_model_probs, axis=0)  # shape: (샘플 수, 클래스 수)
    predicted_labels = np.argmax(ensemble_probs, axis=1)

    print(f"추론 완료: {len(predicted_labels):,}개 예측")

    # 원본 df_test에 pred 컬럼 추가
    df_test["pred"] = predicted_labels

    print(f"\ndf_test에 pred 컬럼이 추가되었습니다. 형태: {df_test.shape}")

    # ========================================
    # 📈 결과 분석
    # ========================================

    print(f"\n{'='*60}")
    print("📈 예측 결과 분석")
    print(f"{'='*60}")

    unique_predictions, counts = np.unique(predicted_labels, return_counts=True)
    print("\n클래스별 예측 분포:")
    for pred, count in zip(unique_predictions, counts):
        percentage = (count / len(predicted_labels)) * 100
        class_name = LABEL_MAPPING.get(pred, f"클래스 {pred}")
        print(f"   {class_name} ({pred}): {count:,}개 ({percentage:.1f}%)")

    # 예측 확신도 분석
    confidence_scores = np.max(ensemble_probs, axis=1)
    print(f"\n📊 앙상블 예측 확신도 통계:")
    print(f"   평균 확신도: {confidence_scores.mean():.4f}")
    print(f"   최소 확신도: {confidence_scores.min():.4f}")
    print(f"   최대 확신도: {confidence_scores.max():.4f}")
    print(f"   중간값: {np.median(confidence_scores):.4f}")

    # 낮은 확신도 샘플 확인
    low_confidence_threshold = 0.4
    low_confidence_count = np.sum(confidence_scores < low_confidence_threshold)
    print(f"   확신도 < {low_confidence_threshold}: {low_confidence_count}개 ({low_confidence_count/len(confidence_scores)*100:.1f}%)")

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("✅ Exponential 가중 앙상블 추론 완료!")
    print(f"{'='*60}")

    # ========================================
    # 💾 제출 파일 생성
    # ========================================

    print(f"\n{'='*60}")
    print("💾 제출 파일 생성 중...")
    print(f"{'='*60}")




def main():
    parser = argparse.ArgumentParser(description='Train Script with Ensemble Option')
    parser.add_argument('-c', '--config', default='./configs/config.json', type=str, help='config file path')
    parser.add_argument('-m', '--mode', default='single', choices=['single', 'ensemble', 's', 'e'], help='training mode')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='GPU indices to enable')
    args = parser.parse_args()


    device = setup_device(args)

    # CLI로 lr, batch size 조정
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(parser, options)

    seed = config['trainer']['args'].get('random_state', 42)
    set_seed(seed)

    mode = config.args.mode.lower()

    if mode in ['ensemble', 'e']:
        trained_models, tokenizers, model_accuracies = ensemble_train(config)
        inference_ensemble(config, trained_models, tokenizers, model_accuracies)
    else:
        train_model(config)



if __name__ == '__main__':
    main()
