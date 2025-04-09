from playwright.sync_api import sync_playwright
import json
import time
import re
import os

def clean_text(text):
    """텍스트 전처리 함수"""
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    # 특수문자 처리
    text = re.sub(r'[\u200b\xa0]', ' ', text)
    return text

def extract_sections(text_content):
    """문서의 섹션별 내용을 추출"""
    sections = []
    current_section = {"title": "", "content": []}
    
    lines = text_content.split('\n')
    for line in lines:
        line = clean_text(line)
        if not line:
            continue
            
        # 새로운 섹션의 시작으로 보이는 패턴
        if re.match(r'^제\s*\d+\s*장|^제\s*\d+\s*조|^[0-9]+\.\s+|^[가-힣]+\s*[0-9]+\.', line):
            if current_section["title"]:
                sections.append(current_section.copy())
            current_section = {"title": line, "content": []}
        else:
            if current_section["title"]:
                current_section["content"].append(line)
            else:
                current_section["title"] = "서문"
                current_section["content"].append(line)
    
    if current_section["title"]:
        sections.append(current_section)
    
    return sections

def create_training_data(sections):
    """파인튜닝용 학습 데이터 생성"""
    training_data = []
    
    # 섹션별로 다양한 형태의 학습 데이터 생성
    for section in sections:
        # 기본 섹션 이해
        entry = {
            "instruction": f"다음 회계기준 내용을 이해하고 설명해주세요:\n{section['title']}",
            "input": "\n".join(section['content']),
            "output": f"{section['title']}에 대한 내용을 이해했습니다. 이 섹션은 회계기준의 중요한 부분을 다루고 있으며, 구체적인 질문에 답변드릴 수 있습니다."
        }
        training_data.append(entry)
        
        # 섹션 내용 요약
        if len(section['content']) > 2:
            entry = {
                "instruction": f"다음 회계기준 내용을 간단히 요약해주세요:\n{section['title']}",
                "input": "\n".join(section['content']),
                "output": f"{section['title']}의 주요 내용을 요약하면 다음과 같습니다. 이 섹션에서는 회계 처리의 기본 원칙과 구체적인 지침을 제공합니다."
            }
            training_data.append(entry)
        
        # 실무 적용 예시
        if len(section['content']) > 3:
            entry = {
                "instruction": f"다음 회계기준을 실무에 적용할 때의 주요 고려사항은 무엇인가요?\n{section['title']}",
                "input": "\n".join(section['content']),
                "output": f"{section['title']}을 실무에 적용할 때는 다음 사항들을 고려해야 합니다. 구체적인 상황에 따라 적절한 판단이 필요합니다."
            }
            training_data.append(entry)
    
    return training_data

def scrape_content():
    # 현재 스크립트의 디렉토리 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 파일명 설정 (확장자 제외)
    base_filename = os.path.splitext(os.path.basename(__file__))[0]
    
    with sync_playwright() as p:
        try:
            print("브라우저 초기화 중...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            print("웹페이지에 접속 중...")
            page.goto("https://db.kasb.or.kr/s/2/J5JVSV?selected=")
            
            # 페이지 로딩 대기
            print("콘텐츠 로딩 중...")
            page.wait_for_load_state('networkidle')
            time.sleep(5)  # 추가 대기 시간
            
            # 제목과 내용 추출
            title = page.title()
            
            # 실제 콘텐츠가 있는 영역 선택 (선택자는 실제 웹페이지 구조에 맞게 수정 필요)
            content = page.evaluate('''() => {
                const contentElement = document.querySelector('.content-area') || document.body;
                return contentElement.innerText;
            }''')
            
            print(f"제목: {title}")
            print("내용 추출 완료")
            
            # 섹션별로 내용 추출
            sections = extract_sections(content)
            print(f"총 {len(sections)}개 섹션 추출 완료")
            
            # 학습 데이터 생성
            training_data = create_training_data(sections)
            print(f"총 {len(training_data)}개의 학습 데이터 생성")
            
            # JSON 파일로 저장
            json_path = os.path.join(script_dir, f"{base_filename}.json")
            print("데이터를 JSON 파일로 저장 중...")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            print("스크래핑 및 JSON 파일 생성 완료!")
            print(f"저장된 파일: {json_path}")
            
            # 원본 데이터도 저장
            html_path = os.path.join(script_dir, f"{base_filename}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(page.content())
            print(f"원본 HTML 저장 완료: {html_path}")
            
            return training_data
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
        finally:
            if 'browser' in locals():
                browser.close()

if __name__ == "__main__":
    scrape_content() 