#datafile중 json 파일을 이용해서 파인튜닝

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)
from datasets import Dataset
import os
from torch.cuda import empty_cache
import gc
import time
from datetime import datetime
import psutil
from transformers.integrations import TensorBoardCallback
from transformers import TrainerCallback
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class CustomCallback(TrainerCallback):
    def __init__(self):
        self.training_start = time.time()
        self.step_start = time.time()
    
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\n{'='*50}")
        print(f"파인튜닝 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
            print(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        print(f"{'='*50}\n")
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.step_start
        total_time = time.time() - self.training_start
        
        if state.global_step % 10 == 0:  # 10스텝마다 출력
            print(f"\n스텝 {state.global_step}")
            print(f"현재 스텝 처리 시간: {step_time:.2f}초")
            print(f"총 학습 시간: {total_time/60:.2f}분")
            if torch.cuda.is_available():
                print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            print(f"CPU 사용률: {psutil.cpu_percent()}%")
            print(f"메모리 사용률: {psutil.virtual_memory().percent}%\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.training_start
        print(f"\n{'='*50}")
        print(f"파인튜닝 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 학습 시간: {total_time/60:.2f}분")
        print(f"총 스텝 수: {state.global_step}")
        if torch.cuda.is_available():
            print(f"최종 GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        print(f"{'='*50}\n")

# 모델 및 토크나이저 설정
MODEL_NAME = "/app/models/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"  # 로컬에 설치된 모델 경로
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuned_gemma_accounting")

def clear_memory():
    """메모리 정리 함수"""
    gc.collect()
    empty_cache()
    
def load_data():
    """학습 데이터 로드 및 전처리 함수
    - JSON 파일에서 회계 기준 데이터를 로드
    - 데이터를 학습에 적합한 형식으로 변환
    - Dataset 객체로 변환하여 반환
    """
    print("현재 작업 디렉토리:", os.getcwd())
    try:
        # JSON 파일 경로 설정 및 로드
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, '제2장 재무제표의작성과표시1.json')
        print(f"JSON 파일 경로: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다. 현재 디렉토리에서 상대 경로를 확인해주세요.\n에러: {e}")
        # 디버깅을 위한 디렉토리 내용 출력
        print("\n현재 디렉토리 파일 목록:")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for root, dirs, files in os.walk(script_dir):
            print(f"\n디렉토리: {root}")
            for f in files:
                print(f"- {f}")
        raise
    except json.JSONDecodeError:
        print("JSON 파일 형식이 올바르지 않습니다.")
        raise
    
    # 데이터 포맷팅: JSON 데이터를 학습용 텍스트로 변환
    formatted_data = []
    for item in data:
        # 각 항목을 구조화된 텍스트로 변환
        text = f"제목: {item.get('title', '')}\n"
        text += f"내용: {item.get('content', '')}\n"
        if 'subcontent' in item:
            text += f"세부내용: {item['subcontent']}\n"
        
        formatted_data.append({
            "input_ids": text,
            "labels": text
        })
    
    print(f"데이터 로드 완료: {len(formatted_data)}개의 학습 데이터")
    return Dataset.from_list(formatted_data)  # HuggingFace Dataset 형식으로 변환

def prepare_model_and_tokenizer():
    """모델과 토크나이저 준비 함수
    - 4비트 양자화 설정
    - KoBERT 토크나이저 로드 및 설정
    - Gemma 모델 로드 및 최적화 (8GB GPU 메모리 최적화)
    """
    try:
        print("\n모델 및 토크나이저 준비 중...")
        start_time = time.time()
        
        # KoBERT 토크나이저 로드 및 설정
        print("한국어 토크나이저 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            "skt/kobert-base-v1",
            use_fast=True,
            trust_remote_code=True
        )
        
        # 특수 토큰 설정
        special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]"
        }
        tokenizer.add_special_tokens(special_tokens)
        print("토크나이저 로딩 완료")
        
        clear_memory()
        
        # 4비트 양자화 설정 (메모리 최적화)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        print("모델 로딩 중...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={
                "": "cuda:0"
            },
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 토크나이저 크기에 맞게 임베딩 레이어 조정
        model.resize_token_embeddings(len(tokenizer))
        
        # PEFT를 위한 모델 준비
        model = prepare_model_for_kbit_training(model)
        
        # LoRA 설정
        config = LoraConfig(
            r=16,                     # LoRA의 랭크
            lora_alpha=32,            # LoRA의 스케일링 파라미터
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 적용할 레이어
            lora_dropout=0.05,        # LoRA 드롭아웃 비율
            bias="none",              # 바이어스 학습 여부
            task_type="CAUSAL_LM"     # 태스크 타입
        )
        
        # LoRA 어댑터 추가
        model = get_peft_model(model, config)
        
        # 그래디언트 체크포인팅 활성화
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        # 학습 가능한 파라미터 정보 출력
        print("\n학습 가능한 파라미터:")
        model.print_trainable_parameters()
        
        print(f"모델 로딩 완료 (소요시간: {time.time()-start_time:.2f}초)")
        if torch.cuda.is_available():
            print(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"모델 또는 토크나이저 로딩 중 오류 발생: {str(e)}")
        raise

def train():
    """메인 학습 함수 (8GB GPU 메모리 최적화)"""
    try:
        print("\n파인튜닝 준비 시작...")
        clear_memory()
        
        # CUDA 가용성 확인
        if torch.cuda.is_available():
            print(f"CUDA 사용 가능 - GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        
        # 학습 데이터 준비
        dataset = load_data()
        
        # 모델과 토크나이저 로드
        model, tokenizer = prepare_model_and_tokenizer()
        
        # 학습 파라미터 설정 (8GB GPU 메모리 최적화)
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,
            learning_rate=5e-6,
            weight_decay=0.01,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=1,
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            max_grad_norm=0.3,
            logging_dir=os.path.join(OUTPUT_DIR, "logs"),
            warmup_steps=50,
            group_by_length=True,
            report_to=["tensorboard"],
            remove_unused_columns=False
        )
        
        # 데이터 전처리 함수
        def preprocess_function(examples):
            # 토크나이저를 사용하여 텍스트를 토큰화
            model_inputs = tokenizer(
                examples["input_ids"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            model_inputs["labels"] = tokenizer(
                examples["labels"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )["input_ids"]
            return model_inputs
        
        # 데이터셋 전처리
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 데이터 콜레이터 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 모니터링 콜백 설정
        callbacks = [CustomCallback(), TensorBoardCallback()]
        
        # 트레이너 초기화 및 학습 실행
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks
        )
        
        print("\n파인튜닝 시작...")
        trainer.train()
        
        # 학습된 모델 저장
        print("\n모델 저장 중...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"모델 저장 완료: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n학습 중 오류 발생: {str(e)}")
        raise
    finally:
        clear_memory()

if __name__ == "__main__":
    print("파인튜닝 스크립트 시작")
    train()  # 학습 실행 
