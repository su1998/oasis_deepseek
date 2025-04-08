# 최초설치후 테스트


from transformers import pipeline
import torch

def main():
    # CUDA 확인
    print("1. CUDA 확인")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 파이프라인 초기화
    print("2. 파이프라인 초기화")
    print("모델 로딩 중...")
    
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-12b-it",
        torch_dtype=torch.bfloat16,
        model_kwargs={"device_map": "auto"}
    )
    print("초기화 완료!\n")

    # 테스트 질문
    question = "인공지능의 발전이 우리 사회에 미치는 영향에 대해 설명해주세요."
    
    print("3. 테스트 시작")
    print(f"질문: {question}\n")
    
    # 프롬프트 구성
    prompt = f"""다음 질문에 한국어로 자세히 답변해주세요:
질문: {question}
답변: """

    # 텍스트 생성
    print("답변 생성 중...")
    result = pipe(
        prompt,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1
    )
    
    # 결과 출력
    print("\n생성된 답변:")
    print("-" * 50)
    print(result[0]['generated_text'])
    print("-" * 50)

    # GPU 메모리 사용량
    if torch.cuda.is_available():
        print(f"\nGPU 메모리 사용량:")
        print(f"할당된 메모리: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"캐시된 메모리: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

if __name__ == "__main__":
    main()





