import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


# 모델 및 토크나이저 로드
model_name = 'irene93/Llama-3-deidentifier'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 모델과 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = torch.nn.DataParallel(model).to(device)

def analyze_news(user_content: str) -> str:
    """
    주어진 뉴스 콘텐츠를 분석하여 JSON 형식으로 결과를 반환합니다.
    """
    # 시스템 메시지와 사용자 메시지 설정
    messages = [
        {"role": "system", "content": "당신은 개인정보를 감춰주는 로봇입니다.\n\n## 지시 사항 ##\n1.주어진 대화에서 사람이름을 [PERSON1], [PERSON2] 등으로 등장 순서에 따라 대체하고, 동일한 이름이 반복될 경우 같은 대치어를 사용합니다.\n2.연락처, 이메일, 주소 , 계좌번호도 각각 [CONTACT1], [CONTACT2] 등, [EMAIL1],[EMAIL2] 등, [ADDRESS1],[ADDRESS2]등 , [ACCOUNT1], [ACCOUNT2] 등 으로 대치하고 동일한 정보가 반복되는 경우에는 같은 대치어를 사용합니다.\n3.대치어를 작성할때 글머리 기호나, 나열식 방식을 쓰지말고 평문으로 이어서 쓰십시오 \n4.위 규칙은 대화 전체에 걸쳐 일관되게 적용합니다. \n당신이 개인정보를 감출 대화내역입니다."},
        {"role": "user", "content": user_content}
    ]

    # 입력 데이터 생성
    input_text = f"{messages[0]['content']}\n\n본문: {messages[1]['content']}\n결과:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)


    # 종료 토큰 설정
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

    # 텍스트 생성 (추론)
    with torch.no_grad():
        outputs = model.module.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=False,
    )

    # 출력 디코딩
    response = outputs[0][input_ids.shape[-1]:]
    output_text = tokenizer.decode(response, skip_special_tokens=True)

    return output_text

def main():
    """
    CLI를 통해 사용자 입력을 받고, 결과를 출력합니다.
    """
    parser = argparse.ArgumentParser(description="대화에서 개인정보 식별")
    parser.add_argument('--input', type=str, required=True, help='식별하고자 하는 대화 (text)')

    args = parser.parse_args()
    user_content = args.input

    result = analyze_news(user_content)
    print(result)

if __name__ == "__main__":
    main()
