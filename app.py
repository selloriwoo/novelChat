import plotly.graph_objects as go # Plotly import 추가
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
import pandas as pd
import numpy as np
import re
import os
import chardet
import toml
import json
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# .env 파일의 경로를 지정합니다.
dotenv_path = "../novelChat/env.txt"

# 지정된 경로의 .env 파일을 로드합니다.
load_dotenv(dotenv_path=dotenv_path)

# OPENAI_API_KEY 환경 변수를 읽어옵니다.
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")
#모델 연결
llm_metadata = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    openai_api_key = openai_api_key
)
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Streamlit에서 소설 파일 업로드
uploaded_files = st.file_uploader(
    "소설 텍스트 파일을 업로드하세요 (여러 권 합치려면 여러 파일 업로드)",
    type=['txt'],
    accept_multiple_files=True
)
# CAPS 분석을 위한 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3500,
    chunk_overlap=400,
    separators=["\n\n", ".", "!", "?", "\n"]
)

novelFullText = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        stringio = uploaded_file.read().decode('utf-8')  # 텍스트 파일이 UTF-8 인코딩이라고 가정
        novelFullText += stringio
    text_chunks = text_splitter.split_text(novelFullText)
    st.success(f"총 {len(uploaded_files)}개의 파일이 업로드되어 분석에 사용됩니다.")
else:
    st.warning("먼저 소설 텍스트(.txt) 파일을 업로드하세요. 예: 해리포터-마법사의돌-1권.txt")
    st.stop()




model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key, max_output_tokens=7000)
prompt1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "등장인물에 이입해서 답변해주는 ai",
        ),
        # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"), # 사용자 입력을 변수로 사용
    ]

)
runnable1 = prompt1 | model1 # 프롬프트와 모델을 연결하여 runnable 객체 생성



store = {} # 세션 기록을 저장할 딕셔너리

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 세션 ID에 해당하는 대화 기록이 저장소에 없으면 새로운 ChatMessageHistory를 생성합니다.
    print(session_id)
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # 세션 ID에 해당하는 대화 기록을 반환합니다.
    return store[session_id]


# 체인에 대화 기록 기능을 추가한 RunnableWithMessageHistory 객체를 생성합니다.
with_message_history1 = RunnableWithMessageHistory(
    runnable1,
    get_session_history,
    # 입력 메시지의 키를 "input"으로 설정합니다.(생략시 Message 객체로 입력)
    input_messages_key="input",
    # 출력 메시지의 키를 "output_message"로 설정합니다. (생략시 Message 객체로 출력)
    # output_messages_key="output_message",
    history_messages_key="chat_history" # 기록 메시지의 키
)

def load_hexaco_questions(filepath="./big5Question.txt"):
    """
    Loads HEXACO questions from a text file into a dictionary.
    텍스트 파일에서 HEXACO 문항을 딕셔너리로 로드하는 함수.

    Args:
        filepath (str): The path to the text file containing the questions.
        문항이 포함된 텍스트 파일의 경로.

    Returns:
        dict: A dictionary where keys are question numbers (int) and values are question strings.
        키는 문항 번호(정수), 값은 문항 문자열인 딕셔너리.
    """
    hexaco_questions = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip() # Remove leading/trailing whitespace 선행/후행 공백 제거
                if line: # Ensure the line is not empty 라인이 비어있지 않은지 확인
                    # 문항 번호와 질문 내용을 분리 (탭 또는 '|' 등으로 구분되어 있다고 가정)
                    # 여기서는 숫자 뒤에 바로 질문이 오는 형태를 파싱하기 위해 정규 표현식을 사용합니다.
                    match = re.match(r'(\d+)\s*(.*)', line)
                    if match:
                        try:
                            question_number = int(match.group(1))
                            question_text = match.group(2).strip()
                            hexaco_questions[question_number] = question_text[1:]
                        except ValueError:
                            st.warning(f"경고: 라인을 파싱할 수 없습니다: {line} (유효하지 않은 문항 번호)")
                    else:
                        st.warning(f"경고: 형식이 잘못된 라인을 건너뜁니다: {line}")
    except FileNotFoundError:
        st.error(f"오류: '{filepath}' 파일을 찾을 수 없습니다. 문항 파일을 올바른 위치에 저장했는지 확인해주세요.")
        st.info("참고: Korean_self100.doc 파일 내용을 복사하여 hexaco_questions.txt 파일로 저장해야 합니다.")
    except Exception as e:
        st.error(f"문항 로딩 중 오류가 발생했습니다: {e}")
    return hexaco_questions


# HEXACO 문항 로드 (Streamlit 앱 실행 시 한번만 로드되도록 캐싱)
@st.cache_data
def cached_load_hexaco_questions(filepath):
    return load_hexaco_questions(filepath)


HEXACO_QUESTIONS = cached_load_hexaco_questions("../novelChat/big5Question.txt")

# 로드된 문항 확인
if HEXACO_QUESTIONS:
    st.success(f"HEXACO 문항 {len(HEXACO_QUESTIONS)}개 로드 완료.")
else:
    st.warning("문항 로드에 실패하여 설문 진행이 어려울 수 있습니다.")

# --- 1. HEXACO 주요 요인 및 하위 척도별 문항 번호 정의 (공식 채점 키 기반) ---
# ScoringKeys_100.pdf 파일에서 추출한 정확한 채점 키를 사용합니다.
# 'items' 리스트에는 해당 하위 척도에 속하는 모든 문항 번호가 포함됩니다.
# 'reverse_items' 리스트에는 해당 하위 척도의 문항 중 역채점해야 할 문항 번호가 포함됩니다.
# Altruism은 주요 6가지 요인 점수 계산 시에는 포함되지 않습니다.

HEXACO_SCORING_KEY_FACETS = {
    'Honesty-Humility': {
        'Sincerity': {'items': [6, 30, 54, 78], 'reverse_items': [6, 54]}, # 6R, 30, 54R, 78
        'Fairness': {'items': [12, 36, 60, 84], 'reverse_items': [12, 36, 84]}, # 12R, 36R, 60, 84R
        'Greed-Avoidance': {'items': [18, 42, 66, 90], 'reverse_items': [42, 66, 90]}, # 18, 42R, 66R, 90R
        'Modesty': {'items': [24, 48, 72, 96], 'reverse_items': [72, 96]} # 24, 48, 72R, 96R
    },
    'Emotionality': { # Big 5의 신경성(Neuroticism)과 유사
        'Fearfulness': {'items': [5, 29, 53, 77], 'reverse_items': [29, 77]}, # 5, 29R, 53, 77R
        'Anxiety': {'items': [11, 35, 59, 83], 'reverse_items': [35, 59]}, # 11, 35R, 59R, 83
        'Dependence': {'items': [17, 41, 65, 89], 'reverse_items': [41, 89]}, # 17, 41R, 65, 89R
        'Sentimentality': {'items': [23, 47, 71, 95], 'reverse_items': [95]} # 23, 47, 71, 95R
    },
    'Extraversion': {
        'Social Self-Esteem': {'items': [4, 28, 52, 76], 'reverse_items': [52, 76]}, # 4, 28, 52R, 76R
        'Social Boldness': {'items': [10, 34, 58, 82], 'reverse_items': [10, 82]}, # 10R, 34, 58, 82R
        'Sociability': {'items': [16, 40, 64, 88], 'reverse_items': [16]}, # 16R, 40, 64, 88
        'Liveliness': {'items': [22, 46, 70, 94], 'reverse_items': [70, 94]} # 22, 46, 70R, 94R
    },
    'Agreeableness': {
        'Forgiveness': {'items': [3, 27, 51, 75], 'reverse_items': [51, 75]}, # 3, 27, 51R, 75R
        'Gentleness': {'items': [9, 33, 57, 81], 'reverse_items': [9]}, # 9R, 33, 57, 81
        'Flexibility': {'items': [15, 39, 63, 87], 'reverse_items': [15, 63, 87]}, # 15R, 39, 63R, 87R
        'Patience': {'items': [21, 45, 69, 93], 'reverse_items': [21, 93]} # 21R, 45, 69, 93R
    },
    'Conscientiousness': {
        'Organization': {'items': [2, 26, 50, 74], 'reverse_items': [50, 74]}, # 2, 26, 50R, 74R
        'Diligence': {'items': [8, 32, 56, 80], 'reverse_items': [56, 80]}, # 8, 32, 56R, 80R
        'Perfectionism': {'items': [14, 38, 62, 86], 'reverse_items': [38]}, # 14, 38R, 62, 86
        'Prudence': {'items': [20, 44, 68, 92], 'reverse_items': [20, 44, 92]} # 20R, 44R, 68, 92R
    },
    'Openness to Experience': {
        'Aesthetic Appreciation': {'items': [1, 25, 49, 73], 'reverse_items': [1, 25]}, # 1R, 25R, 49, 73
        'Inquisitiveness': {'items': [7, 31, 55, 79], 'reverse_items': [55, 79]}, # 7, 31, 55R, 79R
        'Creativity': {'items': [13, 37, 61, 85], 'reverse_items': [13, 85]}, # 13R, 37, 61, 85R
        'Unconventionality': {'items': [19, 43, 67, 91], 'reverse_items': [19, 91]} # 19R, 43, 67, 91R
    }
}

# HEXACO 요인 목록 (출력 순서 지정)
HEXACO_FACTORS_ORDER = [
    'Honesty-Humility',
    'Emotionality',
    'Extraversion',
    'Agreeableness',
    'Conscientiousness',
    'Openness to Experience'
]

# Altruism 척도 (주요 요인 점수 계산 시에는 포함되지 않음)
ALTRUISM_KEY = {
    'items': [97, 98, 99, 100],
    'reverse_items': [99, 100] # 97, 98, 99R, 100R
}

def get_characters_list():
        responseChar = with_message_history1.invoke(  # 이 부분은 실제 대화형 AI 시스템과 연동될 때 활성화
            {"input": HumanMessage(content=novelFullText + """... 여기서 등장인물의 행동과 성격이 파악이 원활히 가능한 등장인물들을 나열 해 줘. 절대적으로 다음 답변형식 대로만 답 해 줘. 답변 형식: 등장인물1, 등장인물2, 등장인물3""")},
            config={"configurable": {"session_id": "aazz1"}})
        response_str = responseChar.content  # 예: "해리 포터, 론 위즐리, 헤르미온느 그레인저, ..."

        # 문자열을 리스트로 변환 (쉼표, 공백 기준으로 분리)
        character_list = [name.strip() for name in response_str.split(",") if name.strip()]

        return character_list

# --- 2. 사용자 응답 입력 함수 ---
def get_user_responses(questions_data):
    """
    사용자로부터 100개의 설문 문항 응답(1~5점)을 입력받는 함수.
    각 문항의 내용을 함께 보여줍니다.
    """
    # 딕셔너리 형태의 질문 데이터를 메시지에 포함할 수 있는 문자열로 변환합니다.
    # 각 질문 번호와 내용을 함께 포맷팅하여 문자열을 만듭니다.
    formatted_questions = ""
    for q_num, q_text in questions_data.items(): # .items()를 사용하여 키와 값을 모두 가져옵니다.
        formatted_questions += f"{q_num}. {q_text}\n" # f-string을 사용하여 '번호. 질문내용' 형식으로 만듭니다.

    responses = {}

    stored=get_session_history("abcd1234")
    response1= with_message_history1.invoke( # 이 부분은 실제 대화형 AI 시스템과 연동될 때 활성화
        {"input": HumanMessage(content=novelFullText+"""... 여기까지 소설 내용이야. 너는 소설 내용을 보고 등장인물인 '론 위즐리'라고 생각하고 각 질문에 대답할거야. 소설 내용 근거로 '론 위즐리' 입장에서 다음 각 번호 질문에 다음 기준으로 점수 답해 줘. 1='전혀 그렇지 않다.' 2='그렇지 않은편이다.' 3='중간정도' 4='그런편이다' 5='매우 그렇다'. 이 기준으로도 모르겠으면 1에서 5까지 자연수로 5에 가까울 수록 더욱더 해당 질문에 동의 하는편으로 기준 생각 해 줘.
        답변형식은 다른 말 하지말고 1. 1, 2. 2, 3. 5, 4. 4, 5. 3 이런식으로 '번호.점수'로 나열해 줘."""+formatted_questions)},
        config={"configurable": {"session_id": "abcd1234"}},
    )


    # print(response1.content)

    response_content_string = response1.content # 실제 응답 대신 더미 데이터 사용

    individual_responses = response_content_string.strip().split('\n')
    a=0
    for item in individual_responses:
        parts = item.split(". ")
        a+=1

        if len(parts) == 2:
            try:
                q_num = int(parts[0]) # 첫 번째 부분(인덱스 0)이 질문 번호
                score = int(parts[1]) # 두 번째 부분(인덱스 1)이 점수
                responses[q_num] = score # 딕셔너리에 '질문 번호: 점수' 형태로 저장
            except ValueError:
                print(f"경고: 유효하지 않은 형식입니다 (숫자 변환 실패): {item, a}")
        else:
            print(f"경고: 예상치 못한 응답 형식입니다 (스킵됨): {item, a}")

    return responses


# --- 3. HEXACO 점수 계산 함수 (주요 요인 및 하위 척도 모두 계산) ---
def calculate_hexaco_scores(responses, scoring_key_facets, factors_order, altruism_key):
    """
    HEXACO-PI-R 주요 요인 및 하위 척도 점수를 계산하는 함수.
    Args:
        responses (dict): 문항 번호(int)를 키로 하고 응답 점수(int)를 값으로 하는 딕셔너리.
        scoring_key_facets (dict): 각 HEXACO 주요 요인 및 하위 척도별 문항 및 역채점 정보 딕셔너리.
        factors_order (list): HEXACO 주요 요인의 순서 리스트.
        altruism_key (dict): 알트루이즘 척도의 문항 및 역채점 정보 딕셔너리.

    Returns:
        tuple: (dict: 주요 요인별 평균 점수, dict: 하위 척도별 평균 점수, dict: 알트루이즘 평균 점수)
    """
    # 하위 척도별 점수 저장
    facet_scores_raw = {
        main_factor: {facet: [] for facet in facet_details.keys()}
        for main_factor, facet_details in scoring_key_facets.items()
    }
    # 주요 요인 점수를 계산하기 위한 모든 원시 점수 (하위 척도 점수를 합산하지 않음)
    main_factor_scores_raw = {factor: [] for factor in factors_order}
    # 알트루이즘 점수
    altruism_raw_scores = []

    # 하위 척도별 점수 처리
    for main_factor, facet_details in scoring_key_facets.items():
        for facet_name, details in facet_details.items():
            items_in_facet = details['items']
            reverse_items_in_facet = details['reverse_items']

            for q_num in items_in_facet:
                if q_num not in responses:
                    st.warning(f"경고: 문항 {q_num}에 대한 응답이 누락되었습니다. {main_factor} - {facet_name} 점수 계산에서 제외됩니다.")
                    continue

                response_score = responses[q_num]

                # 역채점 문항 처리 (1-5점 척도 기준: 6 - 원래 점수)
                if q_num in reverse_items_in_facet:
                    processed_score = 6 - response_score
                else:
                    processed_score = response_score

                facet_scores_raw[main_factor][facet_name].append(processed_score)
                main_factor_scores_raw[main_factor].append(processed_score) # 주요 요인 계산을 위해 원시 점수 추가

    # 알트루이즘 척도 점수 처리
    for q_num in altruism_key['items']:
        if q_num not in responses:
            st.warning(f"경고: 문항 {q_num}에 대한 응답이 누락되었습니다. Altruism 점수 계산에서 제외됩니다.")
            continue
        response_score = responses[q_num]
        if q_num in altruism_key['reverse_items']:
            processed_score = 6 - response_score
        else:
            processed_score = response_score
        altruism_raw_scores.append(processed_score)

    # 최종 하위 척도 평균 점수 계산
    final_facet_scores = {}
    for main_factor, facet_details in facet_scores_raw.items():
        final_facet_scores[main_factor] = {}
        for facet_name, scores in facet_details.items():
            if scores:
                final_facet_scores[main_factor][facet_name] = np.mean(scores)
            else:
                final_facet_scores[main_factor][facet_name] = np.nan

    # 최종 주요 요인 평균 점수 계산
    final_main_factor_scores = {}
    for factor in factors_order:
        scores = main_factor_scores_raw[factor]
        if scores:
            final_main_factor_scores[factor] = np.mean(scores)
        else:
            final_main_factor_scores[factor] = np.nan

    # 최종 알트루이즘 평균 점수 계산
    final_altruism_score = np.mean(altruism_raw_scores) if altruism_raw_scores else np.nan

    return final_main_factor_scores, final_facet_scores, final_altruism_score


# --- 4. 표준 집단 통계 데이터 ---
# "Self-Report Form"의 'Total' 열에 있는 M (평균) 및 SD (표준편차) 값입니다.
# Descriptive Statistics and Internal-Consistency Reliabilities of the HEXACO-100 Scales in a College Student Sample의 데이터
# Altruism은 interstitial facet scale입니다.

STANDARD_MEANS_SELF_REPORT = {
    'Honesty-Humility': 3.19, 'Emotionality': 3.43, 'Extraversion': 3.50,
    'Agreeableness': 2.94, 'Conscientiousness': 3.44, 'Openness to Experience': 3.41
}

STANDARD_SDS_SELF_REPORT = {
    'Honesty-Humility': 0.62, 'Emotionality': 0.62, 'Extraversion': 0.57,
    'Agreeableness': 0.58, 'Conscientiousness': 0.56, 'Openness to Experience': 0.60
}

STANDARD_MEANS_SELF_REPORT_FACETS = {
    'Honesty-Humility': {
        'Sincerity': 3.20, 'Fairness': 3.34, 'Greed-Avoidance': 2.72, 'Modesty': 3.49
    },
    'Emotionality': {
        'Fearfulness': 3.06, 'Anxiety': 3.69, 'Dependence': 3.38, 'Sentimentality': 3.58
    },
    'Extraversion': {
        'Social Self-Esteem': 3.85, 'Social Boldness': 3.03, 'Sociability': 3.59, 'Liveliness': 3.52
    },
    'Agreeableness': {
        'Forgiveness': 2.75, 'Gentleness': 3.17, 'Flexibility': 2.74, 'Patience': 3.11
    },
    'Conscientiousness': {
        'Organization': 3.26, 'Diligence': 3.79, 'Perfectionism': 3.50, 'Prudence': 3.18
    },
    'Openness to Experience': {
        'Aesthetic Appreciation': 3.34, 'Inquisitiveness': 3.19, 'Creativity': 3.63, 'Unconventionality': 3.46
    },
    'Altruism': 3.90
}

STANDARD_SDS_SELF_REPORT_FACETS = {
    'Honesty-Humility': {
        'Sincerity': 0.78, 'Fairness': 0.98, 'Greed-Avoidance': 0.98, 'Modesty': 0.78
    },
    'Emotionality': {
        'Fearfulness': 0.89, 'Anxiety': 0.81, 'Dependence': 0.87, 'Sentimentality': 0.80
    },
    'Extraversion': {
        'Social Self-Esteem': 0.68, 'Social Boldness': 0.87, 'Sociability': 0.75, 'Liveliness': 0.77
    },
    'Agreeableness': {
        'Forgiveness': 0.83, 'Gentleness': 0.73, 'Flexibility': 0.72, 'Patience': 0.86
    },
    'Conscientiousness': {
        'Organization': 0.91, 'Diligence': 0.68, 'Perfectionism': 0.78, 'Prudence': 0.75
    },
    'Openness to Experience': {
        'Aesthetic Appreciation': 0.88, 'Inquisitiveness': 0.88, 'Creativity': 0.85, 'Unconventionality': 0.64
    },
    'Altruism': 0.67
}

# --- 점수 범위별 설명 매핑 데이터 ---
SCORE_INTERPRETATIONS = {
    'Honesty-Humility': {
        (4.21, 5.00): "매우 정직하고 겸손하며, 권력·물질·지위에 대한 욕망이 거의 없음. 도덕적 신념이 확고하고 타인을 착취하거나 속이는 일을 강하게 거부함.",
        (3.61, 4.20): "정직성과 겸손함이 높고, 비교적 소박한 가치관을 지님. 윤리적 기준을 따르려는 경향이 강함.",
        (2.81, 3.60): "평균 수준의 도덕성과 겸손성. 상황에 따라 진실을 조정하거나 자기 이익을 추구할 수도 있음.",
        (2.01, 2.80): "자기중심적 성향이 두드러지며, 권력이나 이익을 위해 불공정하거나 조작적인 행동도 감수할 수 있음.",
        (1.00, 2.00): "매우 이기적이고 기만적인 태도를 지니며, 타인을 착취하거나 조종하려는 경향이 강함. 명예욕과 과시욕이 뚜렷함."
    },
    'Sincerity': { # 진실성
        (4.21, 5.00): "타인의 신뢰를 쉽게 얻음. 거짓말을 극도로 꺼리며, 투명성과 도덕적 책임감을 매우 중요시함. 조작적인 행동에 대해 강한 거부감.",
        (3.61, 4.20): "정직하고 솔직하며, 타인을 조작하려는 의도가 거의 없음. 언행이 일치하는 편.",
        (2.81, 3.60): "평균적인 수준의 진실성. 때로는 상황에 맞춰 솔직함을 조절할 수 있음. 타인을 조작하려는 시도는 적음.",
        (2.01, 2.80): "자신의 이익을 위해 타인에게 비진실한 태도를 보이거나 사실을 왜곡할 수 있음. 타인을 조작하려는 경향이 나타남.",
        (1.00, 2.00): "자신의 이익을 위해 타인을 속이거나 조작하는 경향이 매우 강함. 솔직함보다는 기만적인 태도를 자주 보임."
    },
    'Fairness': { # 공정성
        (4.21, 5.00): "사회 정의에 대한 강한 신념을 가짐. 불공정한 상황이나 차별에 민감하게 반응하고, 약자를 옹호함. 모든 사람이 공정한 대우를 받아야 한다고 굳게 믿음.",
        (3.61, 4.20): "공정성과 정의를 중요하게 여기며, 불공정한 대우에 반대함. 원칙을 지키려는 경향이 강함.",
        (2.81, 3.60): "평균 수준의 공정성. 일반적으로 공정함을 지키려 노력하지만, 개인적인 상황에 따라 유연성을 보일 수 있음.",
        (2.01, 2.80): "개인의 이익이나 편의를 위해 규칙이나 원칙을 어길 수 있음. 특정 상황에서는 불공정한 대우를 묵인할 수도 있음.",
        (1.00, 2.00): "자신의 이익을 위해서라면 기꺼이 불공정하거나 비윤리적인 행동을 할 수 있음. 특권 의식이 강하고 공정성에 대한 관심이 매우 낮음."
    },
    'Greed-Avoidance': { # 탐욕 회피
        (4.21, 5.00): "물질적 욕심이 거의 없고, 부나 사치에 무관심함. 소박하고 절제된 삶을 선호하며, 돈이나 권력으로 사람을 평가하지 않음.",
        (3.61, 4.20): "금전적 이득이나 과시욕이 낮은 편. 검소한 생활을 선호하며, 물질적인 것에 크게 연연하지 않음.",
        (2.81, 3.60): "평균 수준의 물질적 욕구. 적절한 수준의 편안함과 안정성을 추구하지만, 탐욕스럽지는 않음.",
        (2.01, 2.80): "물질적 성공이나 부에 대한 욕구가 강함. 더 많은 돈이나 소유를 위해 노력하며, 때로는 과시적인 경향을 보일 수 있음.",
        (1.00, 2.00): "극도의 탐욕과 물질주의적 성향. 부와 권력을 얻기 위해 수단과 방법을 가리지 않으며, 명예욕과 과시욕이 매우 강함."
    },
    'Modesty': { # 겸손성
        (4.21, 5.00): "자신의 능력이나 성취를 과장하지 않고, 타인의 공로를 인정하며 존중함. 자신을 내세우지 않고 조용히 자신의 역할을 수행함.",
        (3.61, 4.20): "겸손한 태도를 지니며, 자신의 강점을 굳이 드러내려 하지 않음. 타인의 의견을 경청하고 존중함.",
        (2.81, 3.60): "평균 수준의 겸손성. 자신의 업적을 인정받고 싶어 하지만, 지나치게 자신을 내세우지는 않음.",
        (2.01, 2.80): "자신을 과대평가하는 경향이 있으며, 타인의 인정과 칭찬을 중요하게 생각함. 때로는 오만하거나 거만하게 비칠 수 있음.",
        (1.00, 2.00): "자신이 가장 뛰어나다고 생각하며, 타인을 무시하거나 경멸하는 태도를 보임. 끊임없이 자신을 과시하고 우월감을 표현함."
    },
    'Emotionality': {
        (4.21, 5.00): "타인의 감정에 매우 민감하고, 연민과 동정심이 깊음. 타인의 고통을 자신의 고통처럼 느끼고 적극적으로 돕고자 함. 감정 표현이 풍부하고 진솔함.",
        (3.61, 4.20): "타인의 감정에 공감하고 이해하려는 노력을 많이 함. 비교적 감정 표현이 자유롭고, 주변 사람들과 감정을 공유하는 것을 편안하게 생각함.",
        (2.81, 3.60): "평균 수준의 감성. 타인의 감정을 인식하고 반응하지만, 깊이 공감하는 데는 한계가 있을 수 있음. 감정 표현이 적절함.",
        (2.01, 2.80): "감정 표현이 서툴고, 타인의 감정에 무감각한 편. 자신의 감정을 억누르거나 표현하지 않으려 함.",
        (1.00, 2.00): "감수성이 매우 낮고, 타인의 감정이나 고통에 무관심함. 차갑고 냉정한 태도를 보이며, 자신의 감정을 거의 드러내지 않음."
    },
    'Fearfulness': { # 두려움
        (4.21, 5.00): "위험한 상황에 대한 예측이 매우 빠르고, 잠재적 위협에 대해 항상 경계심을 가짐. 신중하고 조심스러운 성향이 강함.",
        (3.61, 4.20): "위험을 회피하고 안전을 추구하는 경향이 강함. 신중하게 행동하며, 불확실한 상황을 꺼려 함.",
        (2.81, 3.60): "평균적인 수준의 두려움. 필요한 경우 위험을 감수하기도 하지만, 대체로 안전을 선호함. 위험에 대해 적절히 반응함.",
        (2.01, 2.80): "새로운 경험이나 위험에 대한 두려움이 낮은 편. 모험을 즐기고, 충동적으로 행동할 수 있음.",
        (1.00, 2.00): "위험에 대한 인식이 거의 없고, 무모하거나 충동적인 행동을 자주 함. 결과를 고려하지 않고 행동에 나서는 경향이 강함."
    },
    'Anxiety': { # 불안
        (4.21, 5.00): "작은 일에도 쉽게 걱정하고 불안감을 느끼며, 스트레스에 매우 취약함. 비관적인 생각에 자주 빠지며, 미래에 대한 염려가 많음.",
        (3.61, 4.20): "걱정이 많고, 긴장하거나 불안감을 자주 느낌. 스트레스에 민감하게 반응하며, 쉽게 위축될 수 있음.",
        (2.81, 3.60): "평균적인 수준의 불안. 적절한 수준의 걱정을 하고, 스트레스 상황에서 비교적 잘 대처함.",
        (2.01, 2.80): "대체로 평온하고 안정적이며, 스트레스를 잘 받지 않는 편. 걱정이나 불안감이 적음.",
        (1.00, 2.00): "극도로 차분하고 침착하며, 거의 불안감을 느끼지 않음. 어떤 상황에서도 흔들리지 않는 강한 정신력을 보임."
    },
    'Dependence': { # 의존성
        (4.21, 5.00): "타인의 도움이나 조언을 적극적으로 구하고, 정서적 지지에 대한 욕구가 강함. 혼자서 결정하기 어려워하며, 타인에게 많이 의지함.",
        (3.61, 4.20): "타인의 의견이나 도움을 구하는 편. 독립적인 행동보다는 협력과 지지를 선호함.",
        (2.81, 3.60): "상황에 따라 타인의 도움을 받거나 독립적으로 행동함. 적절한 수준의 의존성을 보임.",
        (2.01, 2.80): "스스로 결정을 내리고 독립적으로 행동하는 것을 선호함. 타인의 도움을 요청하는 것을 불편해할 수 있음.",
        (1.00, 2.00): "매우 독립적이고 자율적이며, 타인에게 의지하는 것을 극도로 싫어함. 자신의 힘으로 모든 것을 해결하려 함."
    },
    'Sentimentality': { # 감상성
        (4.21, 5.00): "예술, 자연, 추억 등에서 깊은 감동을 받으며, 감수성이 매우 풍부함. 아름다움과 서정적인 것에 대한 감탄을 자주 표현함.",
        (3.61, 4.20): "감성적이고 섬세하며, 아름다움이나 슬픔과 같은 감정에 쉽게 몰입함. 예술 작품이나 자연에서 깊은 감동을 받음.",
        (2.81, 3.60): "평균 수준의 감상성. 감정적인 면모를 가지고 있으나, 현실적인 판단과 균형을 이룸.",
        (2.01, 2.80): "감성적 표현이 적고, 현실적이고 실용적인 것에 더 관심을 가짐. 감정적으로 동요하는 경우가 드묾.",
        (1.00, 2.00): "매우 이성적이고 현실적이며, 감정적인 면모가 거의 없음. 감성적인 것에 무관심하고, 비논리적인 감정을 이해하기 어려워함."
    },
    'Extraversion': {
        (4.21, 5.00): "활력이 넘치고, 사교성이 매우 뛰어남. 항상 사람들 속에 있기를 즐기며, 대화의 중심에 서는 것을 좋아함. 새로운 사람들과 쉽게 친해지고 긍정적인 에너지를 발산함.",
        (3.61, 4.20): "사교적이고 활동적이며, 사람들과 어울리는 것을 즐김. 새로운 경험에 대한 개방성이 높아 다양한 활동에 참여하려 함. 긍정적인 감정을 자주 표현함.",
        (2.81, 3.60): "평균 수준의 외향성. 필요에 따라 사교적인 활동에 참여하지만, 혼자만의 시간도 중요하게 생각함. 적절히 활기찬 모습을 보임.",
        (2.01, 2.80): "조용하고 내성적인 편. 소수의 친한 사람들과 깊은 관계를 맺는 것을 선호하며, 대규모 모임보다는 소규모 활동을 즐김. 에너지를 재충전하기 위해 혼자만의 시간이 필요함.",
        (1.00, 2.00): "매우 조용하고 과묵하며, 사교 활동을 극도로 꺼려 함. 혼자 있는 시간을 가장 편안하게 여기며, 외부 자극에 민감하게 반응함."
    },
    'Social Self-Esteem': { # 사회적 자존감
        (4.21, 5.00): "자신감과 자존감이 매우 높고, 사람들 앞에서 주저함이 없음. 자신의 의견을 명확하게 표현하고, 리더십을 발휘하는 것을 즐김.",
        (3.61, 4.20): "자신감이 있고, 사회적 상황에서 편안함을 느낌. 자신의 능력에 대한 긍정적인 인식을 가지고 있음.",
        (2.81, 3.60): "평균적인 사회적 자존감. 자신감을 가지고 행동하지만, 때로는 타인의 시선을 의식할 수 있음.",
        (2.01, 2.80): "사회적 상황에서 다소 불안감을 느끼거나 자신감이 부족함. 사람들 앞에서 주저하는 경향이 있음.",
        (1.00, 2.00): "자신감과 자존감이 매우 낮고, 사람들 앞에서 위축되거나 불안해함. 자신을 부정적으로 평가하는 경향이 강함."
    },
    'Social Boldness': { # 사회적 대담성
        (4.21, 5.00): "매우 대담하고 적극적이며, 새로운 사람들에게 거리낌 없이 다가감. 사회적 주도성이 높고, 어려운 상황에서도 자신감 있게 행동함.",
        (3.61, 4.20): "사회적 상황에서 적극적이고 대담하게 행동함. 새로운 사람들과의 만남을 즐기며, 자신의 의견을 분명히 표현함.",
        (2.81, 3.60): "평균적인 사회적 대담성. 적절한 수준의 적극성을 보이며, 필요한 경우에만 자신을 드러냄.",
        (2.01, 2.80): "낯선 사람이나 새로운 환경에 대한 불안감이 있어 조심스럽게 행동함. 사람들 앞에서 자신을 내세우는 것을 꺼려 함.",
        (1.00, 2.00): "매우 소극적이고 수줍음이 많으며, 사회적 활동을 극도로 피하려 함. 낯선 상황에 대한 두려움이 강하고, 침묵을 선호함."
    },
    'Sociability': { # 사교성
        (4.21, 5.00): "사람들과 교류하는 것을 극도로 즐기며, 외향적인 활동에 적극적으로 참여함. 넓은 인맥을 형성하고, 항상 사람들과 함께하는 것을 선호함.",
        (3.61, 4.20): "사람들과 어울리는 것을 좋아하고, 다양한 사회적 활동에 참여함. 친목을 중시하며, 다른 사람들과 관계를 맺는 데 적극적임.",
        (2.81, 3.60): "평균적인 사교성. 친한 사람들과의 관계를 중요하게 생각하며, 적당한 수준의 사회 활동을 즐김.",
        (2.01, 2.80): "혼자 있는 시간을 선호하며, 사회적 활동에 대한 욕구가 낮은 편. 소수의 친한 사람들과의 관계에 집중함.",
        (1.00, 2.00): "사람들과의 교류를 거의 하지 않으며, 고독한 시간을 즐김. 사회적 관계에 대한 흥미가 낮고, 타인과의 상호작용을 불편해함."
    },
    'Liveliness': { # 활기
        (4.21, 5.00): "에너지가 넘치고, 항상 긍정적이고 유쾌한 분위기를 만듦. 농담을 즐기고, 사람들을 즐겁게 하는 데 탁월함. 활기찬 모습으로 주변 사람들에게 활력을 줌.",
        (3.61, 4.20): "낙천적이고 쾌활하며, 삶에 대한 열정이 높음. 활동적이며, 주변 사람들에게 긍정적인 영향을 줌.",
        (2.81, 3.60): "평균적인 활기. 적당히 즐겁고 긍정적인 모습을 보이지만, 항상 활기 넘치지는 않음.",
        (2.01, 2.80): "비교적 조용하고 침착한 편. 유머 감각을 자주 드러내지 않으며, 감정 표현이 절제되어 있음.",
        (1.00, 2.00): "매우 차분하고 조용하며, 감정 표현이 거의 없음. 유머 감각이 부족하거나 잘 드러나지 않음. 삶에 대한 열정이 낮은 편."
    },
    'Agreeableness': {
        (4.21, 5.00): "매우 온화하고 관대하며, 타인에게 항상 친절하고 배려심이 깊음. 갈등을 회피하고 조화를 추구하며, 협력적인 태도를 보임.",
        (3.61, 4.20): "타인에게 친절하고 협조적이며, 갈등을 피하려 함. 배려심이 많고, 다른 사람의 의견을 존중함.",
        (2.81, 3.60): "평균 수준의 원만함. 상황에 따라 협조적이거나 단호한 태도를 보임. 갈등을 적절히 관리하려 함.",
        (2.01, 2.80): "자신의 의견을 강하게 주장하고, 때로는 논쟁적이거나 고집스러운 모습을 보임. 타인에게 비판적이거나 냉담할 수 있음.",
        (1.00, 2.00): "매우 비판적이고 적대적이며, 타인과의 갈등을 두려워하지 않음. 냉소적이거나 비꼬는 태도를 자주 보임. 협력보다는 경쟁을 선호함."
    },
    'Forgiveness': { # 용서
        (4.21, 5.00): "타인의 실수나 잘못을 쉽게 용서하고, 복수심이 거의 없음. 너그럽고 관대한 마음으로 모든 것을 받아들임. 과거의 부정적인 감정에 얽매이지 않음.",
        (3.61, 4.20): "타인의 잘못을 비교적 쉽게 용서하고, 앙심을 품지 않으려 함. 관계의 회복을 중요하게 생각함.",
        (2.81, 3.60): "평균 수준의 용서. 상황에 따라 용서하거나 앙심을 품을 수 있음. 용서에는 시간이 필요함.",
        (2.01, 2.80): "타인의 잘못을 쉽게 잊지 못하고, 앙심을 품거나 복수심을 가질 수 있음. 과거의 부정적인 경험에 얽매이는 경향이 있음.",
        (1.00, 2.00): "타인의 잘못을 절대로 용서하지 않으려 하며, 복수심이 매우 강함. 작은 실수에도 분노를 느끼고, 원한을 오래도록 기억함."
    },
    'Gentleness': { # 온화함
        (4.21, 5.00): "매우 온화하고 부드러우며, 타인에게 상냥하고 친절함. 공격적인 태도를 보이지 않고, 갈등을 평화롭게 해결하려 함.",
        (3.61, 4.20): "타인에게 부드럽고 친절하게 대하며, 공격적인 행동을 피하려 함. 차분하고 온화한 성향을 지님.",
        (2.81, 3.60): "평균 수준의 온화함. 대체로 친절하지만, 상황에 따라 단호한 태도를 보일 수 있음. 감정을 적절히 조절함.",
        (2.01, 2.80): "때로는 성급하거나 거친 태도를 보일 수 있음. 자신의 의견을 강하게 주장하며, 쉽게 화를 내거나 짜증을 낼 수 있음.",
        (1.00, 2.00): "매우 공격적이고 거친 태도를 지니며, 쉽게 분노하고 짜증을 냄. 타인에게 불친절하거나 비판적인 언행을 자주 함."
    },
    'Flexibility': { # 유연성
        (4.21, 5.00): "생각이 매우 유연하고 개방적이며, 다양한 관점을 존중하고 받아들임. 자신의 의견이 틀렸을 때도 쉽게 인정하고 수정함. 새로운 아이디어에 대한 수용성이 높음.",
        (3.61, 4.20): "의견을 기꺼이 수정하고, 새로운 정보나 관점에 대해 열린 마음을 가짐. 유연하게 생각하며, 변화를 긍정적으로 받아들임.",
        (2.81, 3.60): "평균 수준의 유연성. 대체로 합리적으로 판단하지만, 때로는 자신의 생각에 고집을 부릴 수 있음. 새로운 것에 대한 적응력이 보통임.",
        (2.01, 2.80): "자신의 생각이나 신념을 쉽게 바꾸려 하지 않음. 새로운 아이디어나 변화에 대해 저항적인 태도를 보일 수 있음.",
        (1.00, 2.00): "매우 완고하고 고집이 세며, 자신의 의견이나 신념을 절대적으로 옳다고 믿음. 새로운 정보나 관점을 전혀 받아들이지 않음. 융통성이 매우 부족함."
    },
    'Patience': { # 인내심
        (4.21, 5.00): "어떤 상황에서도 침착하고 인내심이 매우 강함. 좌절이나 어려움 속에서도 평온함을 유지하며, 장기적인 목표를 위해 꾸준히 노력함.",
        (3.61, 4.20): "인내심이 높고, 쉽게 좌절하거나 포기하지 않음. 어려운 상황에서도 끈기 있게 노력하는 편.",
        (2.81, 3.60): "평균 수준의 인내심. 어느 정도의 어려움은 견디지만, 한계를 넘어서면 인내심을 잃을 수 있음.",
        (2.01, 2.80): "쉽게 조바심을 내거나 인내심이 부족한 편. 결과를 빠르게 얻고 싶어 하며, 지루하거나 반복적인 일을 싫어함.",
        (1.00, 2.00): "매우 참을성이 없고, 작은 어려움에도 쉽게 포기함. 즉각적인 만족을 추구하며, 지루함을 견디지 못함."
    },
    'Conscientiousness': {
        (4.21, 5.00): "매우 성실하고 책임감이 강하며, 맡은 일은 완벽하게 해냄. 계획적이고 체계적이며, 항상 최고 수준의 성과를 추구함. 시간 관리에 탁월하고 자기 규율이 매우 강함.",
        (3.61, 4.20): "성실하고 책임감이 높으며, 맡은 일을 꾸준히 수행함. 계획을 세우고 목표 달성을 위해 노력하는 편. 자기 통제력이 좋음.",
        (2.81, 3.60): "평균 수준의 성실성. 필요한 경우에만 계획을 세우고, 업무를 수행함. 자기 통제력이 보통임.",
        (2.01, 2.80): "비교적 자유분방하고 즉흥적인 편. 계획을 세우는 것을 싫어하고, 마감 기한을 지키는 데 어려움을 겪을 수 있음. 책임감이 낮은 편.",
        (1.00, 2.00): "매우 충동적이고 무책임하며, 계획 없이 행동하고 자주 일을 미룸. 의무를 회피하고, 자기 규율이 거의 없음."
    },
    'Organization': { # 조직화
        (4.21, 5.00): "매우 체계적이고 정리정돈을 잘하며, 모든 일을 효율적으로 계획하고 실행함. 주변 환경이 항상 깔끔하고 질서정연함.",
        (3.61, 4.20): "깔끔하고 정돈된 환경을 선호하며, 일을 체계적으로 처리하는 경향이 있음. 계획을 세우고 효율적으로 일하려 함.",
        (2.81, 3.60): "평균 수준의 조직화 능력. 필요에 따라 정리정돈을 하지만, 항상 완벽하지는 않음. 적절한 수준의 계획성을 가짐.",
        (2.01, 2.80): "정리정돈에 관심이 적고, 주변 환경이 다소 어수선할 수 있음. 즉흥적으로 일을 처리하는 것을 선호함.",
        (1.00, 2.00): "매우 무질서하고 혼란스러운 환경을 선호하며, 계획 없이 행동하고 자주 일을 미룸. 정리정돈을 거의 하지 않음."
    },
    'Diligence': { # 근면성
        (4.21, 5.00): "매우 근면하고 성실하며, 어떤 일이든 꾸준히 노력하여 목표를 달성함. 힘든 업무도 마다하지 않고, 탁월한 인내심으로 끝까지 해냄.",
        (3.61, 4.20): "맡은 일을 성실하게 수행하고, 목표를 달성하기 위해 꾸준히 노력함. 책임감이 강하고, 쉽게 포기하지 않음.",
        (2.81, 3.60): "평균 수준의 근면성. 필요한 만큼의 노력을 기울이며, 업무를 적절히 수행함.",
        (2.01, 2.80): "일을 미루거나 게으름을 피우는 경향이 있음. 쉬운 일을 선호하고, 힘든 업무는 피하려 함.",
        (1.00, 2.00): "매우 게으르고 나태하며, 책임감이 거의 없음. 최소한의 노력만 기울이고, 어려운 일은 회피하려 함."
    },
    'Perfectionism': { # 완벽주의
        (4.21, 5.00): "모든 일에 완벽을 추구하며, 작은 실수도 용납하지 않음. 자신과 타인에게 높은 기준을 적용하고, 항상 최고를 지향함.",
        (3.61, 4.20): "높은 기준을 가지고 일하며, 완벽을 추구하려는 경향이 있음. 디테일에 신경 쓰고, 실수를 줄이려 노력함.",
        (2.81, 3.60): "평균 수준의 완벽주의. 실수에 대해 적절히 반응하고, 합리적인 수준에서 최선을 다함.",
        (2.01, 2.80): "실수에 대해 비교적 너그러운 편이며, 완벽보다는 적절한 결과에 만족함. 과정의 효율성을 중시함.",
        (1.00, 2.00): "완벽주의 성향이 거의 없고, 실수나 불완전함에 대해 크게 신경 쓰지 않음. 대충 만족하는 경향이 강함."
    },
    'Prudence': { # 신중성
        (4.21, 5.00): "매우 신중하고 조심스러우며, 모든 행동과 결정에 앞서 충분히 숙고함. 충동적인 행동을 피하고, 위험을 최소화하려 노력함.",
        (3.61, 4.20): "결정하기 전에 충분히 생각하고, 위험을 피하려 함. 신중하게 행동하며, 충동적인 결정을 내리지 않으려 함.",
        (2.81, 3.60): "평균 수준의 신중성. 필요에 따라 신중하게 행동하지만, 때로는 직관적으로 판단하기도 함.",
        (2.01, 2.80): "충동적이고 즉흥적인 경향이 있음. 위험을 감수하는 것을 두려워하지 않으며, 빠르게 결정하는 편.",
        (1.00, 2.00): "매우 무모하고 충동적이며, 결과를 고려하지 않고 행동에 나섬. 위험을 즐기고, 즉흥적인 결정을 선호함."
    },
    'Openness to Experience': { # 경험에 대한 개방성
        (4.21, 5.00): "지적 호기심이 매우 강하고, 학문·철학·사회 문제 등 다양한 주제에 깊은 관심을 가짐. 책, 다큐멘터리, 강의 등을 즐기며, 항상 배우고자 하는 태도. 새로운 개념이나 복잡한 아이디어에 흥미를 느낌.",
        (3.61, 4.20): "평균 이상으로 새로운 지식과 정보에 관심이 있으며, 스스로 학습하거나 탐구하는 걸 즐김. 일상 속 호기심이 많아 다양한 질문을 자주 던짐.",
        (2.81, 3.60): "일상적이고 실용적인 정보에 관심이 있고, 꼭 필요할 때만 학습에 집중. 지적 탐색을 즐기지만 강한 집착은 없음.",
        (2.01, 2.80): "새로운 지식에 대한 욕구가 낮은 편. 복잡한 개념보다는 익숙한 정보나 활동을 선호함. 관심 분야가 제한적.",
        (1.00, 2.00): "학문적·지적 관심이 거의 없고, 새로운 정보에 대한 탐색욕이 부족함. 변화보다는 익숙함을 중시하고, 질문을 덜 던지는 편."
    },
    'Aesthetic Appreciation': { # 심미성
        (4.21, 5.00): "예술, 음악, 자연 등에서 깊은 아름다움을 느끼고, 감수성이 매우 풍부함. 미적 경험을 즐기며, 일상 속에서도 아름다움을 발견하려 함.",
        (3.61, 4.20): "예술과 아름다움에 대한 높은 감수성을 지님. 미술관이나 음악회 관람을 즐기며, 미적 경험을 중요하게 생각함.",
        (2.81, 3.60): "평균 수준의 심미성. 예술이나 자연의 아름다움에 대해 적절히 반응하지만, 깊이 몰입하지는 않음.",
        (2.01, 2.80): "예술이나 미적인 것에 대한 관심이 적고, 실용적이고 기능적인 것을 선호함. 감성적인 경험보다는 이성적인 판단을 중요하게 생각함.",
        (1.00, 2.00): "예술적 감수성이 거의 없고, 미적인 것에 무관심함. 실용적인 측면을 중시하며, 감성적인 경험을 이해하기 어려워함."
    },
    'Inquisitiveness': { # 탐구심
        (4.21, 5.00): "지적 호기심이 매우 강하고, 학문·철학·사회 문제 등 다양한 주제에 깊은 관심을 가짐. 책, 다큐멘터리, 강의 등을 즐기며, 항상 배우고자 하는 태도. 새로운 개념이나 복잡한 아이디어에 흥미를 느낌.",
        (3.61, 4.20): "평균 이상으로 새로운 지식과 정보에 관심이 있으며, 스스로 학습하거나 탐구하는 걸 즐김. 일상 속 호기심이 많아 다양한 질문을 자주 던짐.",
        (2.81, 3.60): "일상적이고 실용적인 정보에 관심이 있고, 꼭 필요할 때만 학습에 집중. 지적 탐색을 즐기지만 강한 집착은 없음.",
        (2.01, 2.80): "새로운 지식에 대한 욕구가 낮은 편. 복잡한 개념보다는 익숙한 정보나 활동을 선호함. 관심 분야가 제한적.",
        (1.00, 2.00): "학문적·지적 관심이 거의 없고, 새로운 정보에 대한 탐색욕이 부족함. 변화보다는 익숙함을 중시하고, 질문을 덜 던지는 편."
    },
    'Creativity': { # 창의성
        (4.21, 5.00): "아이디어가 넘치며, 상상력과 발상이 탁월함. 예술, 문학, 과학 등 다양한 분야에서 독창적인 생각을 해내고 싶어 함. 문제 해결에 있어 창의적인 접근을 선호함.",
        (3.61, 4.20): "새롭고 독창적인 생각을 즐기며, 고정관념에서 벗어나 사고하려 함. 다양한 아이디어를 제시하고, 창의적인 활동에 관심이 많음.",
        (2.81, 3.60): "평균 수준의 창의성. 필요에 따라 새로운 아이디어를 내지만, 항상 독창적인 것은 아님. 현실적인 문제 해결에 집중함.",
        (2.01, 2.80): "새로운 아이디어를 내는 데 어려움을 느끼며, 익숙하고 검증된 방법을 선호함. 상상력보다는 현실적인 것에 초점을 맞춤.",
        (1.00, 2.00): "창의적인 사고를 거의 하지 않으며, 고정관념에 갇혀 있음. 새롭거나 독창적인 것을 싫어하고, 익숙한 방식만을 고집함."
    },
    'Unconventionality': { # 비인습성
        (4.21, 5.00): "매우 독특하고 비전통적인 사고방식을 지니며, 사회적 규범이나 관습에 얽매이지 않음. 개성이 강하고, 자신만의 독특한 삶의 방식을 추구함.",
        (3.61, 4.20): "관습이나 전통에 얽매이지 않고, 자신만의 방식으로 생각하고 행동하려 함. 개성이 강하고, 새로운 것에 대한 호기심이 많음.",
        (2.81, 3.60): "평균 수준의 비인습성. 사회적 규범을 따르지만, 때로는 자신만의 방식을 시도하기도 함. 새로운 것에 대해 개방적임.",
        (2.01, 2.80): "전통적이고 보수적인 사고방식을 선호하며, 변화나 새로운 것에 저항적임. 사회적 규범을 중요하게 생각하고, 안정적인 삶을 추구함.",
        (1.00, 2.00): "매우 보수적이고 전통적인 가치를 중시하며, 변화나 새로운 것을 극도로 싫어함. 규범을 철저히 따르고, 예측 가능한 삶을 선호함."
    },
    'Altruism': { # 이타성
        (4.21, 5.00): "타인의 고통에 깊이 공감하고, 아무런 대가 없이 타인을 돕고자 하는 강한 욕구를 지님. 약자에게 연민을 느끼고, 희생적인 태도를 보임.",
        (3.61, 4.20): "타인에게 친절하고 너그러우며, 어려운 사람을 돕는 데 기꺼이 시간을 할애함. 연민과 동정심이 높은 편.",
        (2.81, 3.60): "평균 수준의 이타성. 필요에 따라 타인을 돕지만, 자신의 이익도 고려함. 적절한 수준의 공감 능력을 가짐.",
        (2.01, 2.80): "타인의 어려움에 대해 크게 신경 쓰지 않으며, 자신의 이익을 우선시하는 경향이 있음. 냉담하거나 무관심할 수 있음.",
        (1.00, 2.00): "매우 이기적이고 냉정하며, 타인의 고통에 무관심함. 자신의 이익을 위해서라면 타인을 이용하거나 해칠 수 있음."
    }
}


def interpret_score(score, trait_name):
    """
    주어진 점수에 해당하는 성격 설명을 반환합니다.
    Args:
        score (float): HEXACO 요인 또는 하위 척도의 점수.
        trait_name (str): 해석할 특성 (예: 'Honesty-Humility', 'Sincerity').
    Returns:
        str: 해당 점수 범위에 대한 설명.
    """
    if pd.isna(score):
        return "점수 계산 불가"

    interpretations = SCORE_INTERPRETATIONS.get(trait_name)
    if not interpretations:
        return "해당 특성에 대한 설명이 없습니다."

    for (lower, upper), description in interpretations.items():
        if lower <= score <= upper:
            return description
    return "점수 범위 외"

# --- 등장인물별 get_user_responses 함수 ---
def get_user_responses_for_character(character_name, questions_data):
    """
    등장인물 이름에 따라 LLM에 답변을 요청하는 함수.
    """
    formatted_questions = ""
    for q_num, q_text in questions_data.items():
        formatted_questions += f"{q_num}. {q_text}\n"
    responses = {}
    session_id = f"hexaco_{character_name}"
    stored = get_session_history(session_id)
    response1 = with_message_history1.invoke(
        {"input": HumanMessage(content=novelFullText +
            f"... 여기까지 소설 내용이야. 너는 소설 내용을 보고 등장인물인 '{character_name}'라고 생각하고 각 질문에 대답할거야. 소설 내용 근거로 '{character_name}' 입장에서 다음 각 번호 질문에 다음 기준으로 점수 답해 줘. 1='전혀 그렇지 않다.' 2='그렇지 않은편이다.' 3='중간정도' 4='그런편이다' 5='매우 그렇다'. 이 기준으로도 모르겠으면 1에서 5까지 자연수로 5에 가까울 수록 더욱더 해당 질문에 동의 하는편으로 기준 생각 해 줘.\n답변형식은 다른 말 하지말고 1. 1, 2. 2, 3. 5, 4. 4, 5. 3 이런식으로 '번호.점수'로 나열해 줘.\n" +
            formatted_questions)},
        config={"configurable": {"session_id": session_id}},
    )
    response_content_string = response1.content
    individual_responses = response_content_string.strip().split('\n')
    for item in individual_responses:
        parts = item.split(". ")
        if len(parts) == 2:
            try:
                q_num = int(parts[0])
                score = int(parts[1])
                responses[q_num] = score
            except ValueError:
                print(f"경고: 유효하지 않은 형식입니다 (숫자 변환 실패): {item}")
        else:
            print(f"경고: 예상치 못한 응답 형식입니다 (스킵됨): {item}")
    return responses

def create_radar_chart(scores, categories, title):
    """
    Plotly를 사용하여 레이더 차트를 생성합니다.
    - 범주(카테고리) 라벨은 흰색
    - 반지름(숫자 눈금) 라벨은 빨간색
    """
    scores = scores + scores[:1]
    categories = categories + categories[:1]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name='등장인물 점수'
            )
        ],
        layout=go.Layout(
            title=go.layout.Title(text=title),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[1, 5],
                    tickfont=dict(color='red'),  # 반지름(숫자) 빨간색
                    tickvals=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                ),
                angularaxis=dict(
                    tickfont=dict(color='white'),  # 카테고리(글자) 흰색
                )
            ),
            showlegend=True
        )
    )
    return fig

def analyze_caps_signatures(
    character_name,
    text_chunks,
    llm_metadata,
    extract_json_from_codeblock,
    st,
    save_to_file=True
):
    """
        소설 텍스트 청크, 인물명, LLM 객체를 받아 CAPS 행동서명을 추출.
        결과는 행동서명 리스트(caps_results)로 반환.

        Args:
            character_name (str): 분석할 등장인물 이름
            text_chunks (list[str]): 소설 텍스트를 분할한 청크 리스트
            llm_metadata: langchain_openai.ChatOpenAI 객체
            extract_json_from_codeblock: 코드블록에서 json을 추출하는 함수
            st: streamlit 모듈 (세션/마크다운 등)
            save_to_file (bool): txt 파일로 결과 저장 여부

        Returns:
            list: CAPS 행동서명 결과 리스트
        """
    # 8-1. 메타데이터 프롬프트
    system_caps_msg = SystemMessage(content="""
                                            당신은 소설 텍스트로부터 메타데이터를 추출하는 분석 도우미입니다.
                                            다음은 소설의 한 부분입니다. 
                                            텍스트를 읽고, 아래 항목을 바탕으로 정보를 **정확한 JSON 형식**으로 추출하세요. 
                                            응답에는 반드시 **JSON만 출력**하십시오. 설명, 해설, 주석은 절대 포함하지 마세요.

                                            [situation 추출 방법]
                                            1. 상황 유형은 DIAMONDS 분류 체계를 기반으로 아래의 상황 유형 목록을 참고하고,각 상위 유형 안에서 적절한 하위 유형을 모두 situation_detail에 추출하세요.
                                            2. 추출된 하위 유형들의 상위 유형을 계산하여, 상위 유형별 비율(%)을 계산하여 situation_weight에 추출하세요.
                                                예) 하위 유형이 Duty에서 3개, Adversity에서 1개 추출되었을 경우
                                                → situation_type: ["Duty", "Adversity"]
                                                → situation_weight: { "Duty": 75, "Adversity": 25 }
                                            3. situation_weight에 포함된 모든 상위 유형 각각에 대해, 그 하위 유형들을 함께 병기하여 situation_name에 추출하세요.
                                                구체적 설명: "상위 유형: 하위 유형1.../ 상위 유형: 하위 유형1... "으로 상위유형과 관련된 하위 유형들을 함께 병기한 상황명.  
                                                            -> 예시 형식: 'Deception: Lying, Social Persona / Duty: Responsibility'


                                            [상황 유형 목록]
                                            # DIAMONDS 하위 유형 사전
                                            DIAMONDS_DETAIL_DICT = {
                                                "Duty": ["Responsibility", "Integrity", "Loyalty", "Norm Compliance", "Self-sacrifice", "Diligence"],
                                                "Intellect": ["Analytical Thinking", "Creativity", "Learning Ability", "Critical Thinking", "Curiosity", "Logical Reasoning"],
                                                "Adversity": ["Resilience", "Patience", "Fear Overcoming", "Stress Management", "Crisis Response", "Frustration Tolerance"],
                                                "Mating": ["Attractiveness", "Mate Preference", "Courtship Behavior", "Jealousy", "Sexual Preference", "Relationship Maintenance"],
                                                "positivity": ["Optimism", "Gratitude", "Hopefulness", "Self-affirmation", "Sense of Humor", "Happiness Pursuit"],
                                                "Negativity": ["Anxiety", "Depression", "Anger", "Self-deprecation", "Cynicism", "Social Alienation"],
                                                "Deception": ["Lying", "Concealment", "Camouflage Strategy", "Duplicity", "Manipulativeness", "Social Persona"],
                                                "Sociality": ["Empathy", "Cooperativeness", "Communication Skill", "Interpersonal Appeal", "Norm Awareness", "Relationship Building"]
                                            }

                                            # 한글 카테고리명
                                            DIAMONDS_KO_LABEL = {
                                                "Duty": "의무", "Intellect": "지성", "Adversity": "역경", "Mating": "교제",
                                                "positivity": "긍정성", "Negativity": "부정성", "Deception": "기만", "Sociality": "사교성"
                                            }

                                            [출력할 메타데이터 항목]
                                            - character: 사건을 주도하거나 직접 겪는 주요 인물 (한 명만)
                                            - situation_type: 인물이 처한 핵심 사건의 상위 유형 (복수 가능)
                                            - situation_detail: 해당 상황의 세부 유형 (복수 가능)
                                            - situation_weight: 상위 유형별 가중치 비율 (%). 비율의 총합은 100이 되도록 조정
                                            - situation_name: **규칙:situation_detail에 추출된 모든 하위 유형을 빠짐없이situation_name에 포함시켜야 합니다. 절대로 일부를 생략해서는 안 됩니다.
                                                ** 형식 "상위 유형: 하위 유형1.../ 상위 유형: 하위 유형1... "으로 상위유형과 관련된 하위 유형들을 함께 병기한 상황명.  
                                                ** 예시 형식: 'Deception: Lying, Social Persona / Duty: Responsibility'
                                            - emotion: 이 인물이 느끼는 핵심 감정들 (복수 가능)
                                            - action: 이 인물이 취한 행동 또는 반응
                                            - belief: 이 인물이 상황을 해석하고 행동하게 만든 신념 또는 가치
                                            - goal: 이 인물이 행동을 통해 달성하려는 의도

                                            [추가 규칙]
                                            - 여러 상황이 명확히 분리될 경우, 각각 JSON 객체로 나누고 리스트 형태로 출력하세요.
                                            - 직접 연결된 단일 상황이라면 하나의 상황으로 묶어도 무방합니다.
                                            - 정보가 명확하지 않다면 어떻게든 비슷한 유형을 판단하고 분류도도록 하십시오.
                                            - 결과는 반드시 JSON 배열(list) 형식으로 출력하세요.
                                            - 등장인물의 이름은 반드시 풀네임으로 말해 주세요.

                                            [출력 예시]
                                            [
                                              {
                                                "character": "해리 포터",
                                                "situation_type": ["Deception", "Duty"],
                                                "situation_detail": ["Lying", "Manipulation", "Responsibility"],
                                                "situation_weight": {"Deception": 66,"Duty": 34},
                                                "situation_name": Deception: Lying, Manipulation / Duty: Responsibility",
                                                "emotion": ["경계심", "책임감"],
                                                "action": "상대의 의도를 의심하고 통제권을 유지하려 함",
                                                "belief": "타인은 나를 속일 수 있으므로 내가 통제해야 한다",
                                                "goal": "상황을 장악하고 위험을 차단하기 위해"
                                              }
                                            ]
                                            """)
    # 8-2. CAPS 행동서명 프롬프트
    system_caps2_msg = SystemMessage(content="""당신은 성격 심리학의 CAPS 이론(Cognitive-Affective Personality System)을 기반으로 
                                             특정 인물이 특정 상황 유형에서 반복적으로 보이는 행동 패턴을 요약하는 분석 전문가입니다.

                                             CAPS 이론은 다음과 같은 전제를 가지고 있습니다:
                                             - 사람은 각기 다르게 상황을 해석하고, 그 해석과 신념, 목표에 기반해 감정과 행동이 형성됩니다.
                                             - 특정 인물은 특정 상황에서 반복적으로 유사한 해석과 행동을 보이는 경향성이 있으며, 이것을 행동 서명이라 부릅니다.

                                             다음은 하나의 인물(character)이 DIAMONDS 8가지 상황 유형 중 **복수의 상황 유형**에 동시에 놓여 있는 메타데이터입니다.  
                                             당신의 임무는 이 데이터를 분석하여 다음 두 가지 정보를 생성하세요:

                                             1. 여러 상황 유형 조합에 해당하는 **행동 서명 1개**
                                             2. 주어진 상황 유형별 가중치(situation_weight)를 JSON으로 그대로 출력

                                             [입력 항목]
                                             - situation_name: 각 상위 유형에 대해 해당하는 하위 유형들을 함께 병기하세요.
                                               형식: 'Deception: Lying, Manipulation / Duty: Responsibility'
                                             - situation_type에 속하는 situation_detail을 나열하여 전달받기기
                                             - emotion: 인물이 느낀 감정들
                                             - action: 인물이 취한 행동
                                             - belief: 행동을 유도한 신념 또는 가치관
                                             - goal: 행동을 통해 달성하려는 목적
                                             - situation_weight: 상황 유형별 가중치 (예: {"Deception": 66, "Duty": 34})

                                             [출력 규칙]
                                             - 반드시 **JSON 배열(list)**로 출력하세요.
                                             - 첫 번째 JSON 객체는 If–Then 형식의 행동 서명입니다.
                                             - 행동 서명 문장의 상황 부분은 **입력으로 받은 situation_name의 내용을 그대로 반영**해야 합니다. **입력된 모든 상위 유형과 그에 해당하는 하위 유형을 빠짐없이 정확히 포함**하여 작성하십시오.
                                             - 두 번째 JSON 객체는 situation_weight 항목은 이번 프롬포트로 추출되는 [상황]에 대하여 산정하여 작성하시오
                                             - 행동 서명 문장은 **상황들이 복합적으로 작용했을 때의 공통된 경향성**을 일반화하여 1~2문장으로 기술하세요.
                                             - 출력물의 상황유형은 '상황유형': '하위유형', '하위유형'.... 의 형식으로 해당하는 하위 내용을 모두 출력하도록 하시오.

                                             [출력 예시]
                                             [
                                              { "만약 인물이 상황을 'Deception: Lying, Manipulation / Duty: Responsibility'이라고 인식한다면,  
                                                 그는 '상대의 의도를 경계하면서도 조직적 통제를 유지하려는 경향이 있다.'" },
                                              { "상황 가중치 비율": {"Deception": 66, "Duty": 34}}
                                             ]

                                             [작성 가이드]
                                             - situation_name은 **입력으로 받은 데이터를 그대로 인용하여 사용**하세요. (단일 또는 다중 카테고리 모두 가능)
                                             - 행동 표현은 추상적 경향성을 담고 있어야 하며, 특정 인물·도구·보상 중심이어선 안 됩니다.
                                             - 복수 상황의 상호작용을 반영한 ‘통합적 태도’나 ‘우선 반응 경향’을 중심으로 서술하세요.
                                             - belief, emotion, goal 간의 인과적 흐름을 고려해 행동 서명 문장을 구성하세요.
                                             """)

    results = []
    for chunk in text_chunks:
        human_msg = HumanMessage(content=f"소설 내용:\n\"\"\"\n{chunk}\n\"\"\"")
        response = llm_metadata.invoke([system_caps_msg, human_msg])
        results.append(response.content)

    parsed = []
    for idx, r in enumerate(results):
        try:
            json_str = extract_json_from_codeblock(r)
            parsed += json.loads(json_str)
        except Exception as e:
            print(f"JSON 파싱 실패 at idx {idx}: {e}")
            print("------ RAW 응답 ------")
            print(r)
    print(f"parsed 개수: {len(parsed)}")
    print(parsed)

    filtered_entries = [e for e in parsed if e["character"] == character_name]
    grouped_by_situation = defaultdict(list)
    caps_results = []
    for entry in filtered_entries:
        for situation in entry.get("situation_type", []):
            grouped_by_situation[situation].append(entry)

    for situation, entries in grouped_by_situation.items():
        entry = entries[0]  # 대표 샘플
        situation_name = entry["situation_name"]
        situation_weight = entry["situation_weight"]
        emotion = entry["emotion"]
        belief = entry["belief"]
        action = entry["action"]
        goal = entry["goal"]

        prompt = f"""다음은 '{character_name}'가 '{situation}' 상황에서 반복적으로 보이는 행동 양식입니다.
                        - 상황 분류(situation_name): {situation_name}
                        - 상황 가중치(situation_weight): {situation_weight}
                        - 감정(emotion): {emotion}
                        - 신념(belief): {belief}
                        - 행동(action): {action}
                        - 목표(goal): {goal}

                        이 정보를 바탕으로 CAPS 이론 기반 If–Then 행동 서명을 생성하세요.
                        """
        human_msg = HumanMessage(content=prompt.strip())
        try:
            response = llm_metadata.invoke([system_caps2_msg, human_msg])
            caps_results.append(response.content.strip())
        except Exception as e:
            st.warning(f"❌ LLM 오류 at situation '{situation}': {e}")

    # 세션 상태에 분석 결과 캐싱!
    st.session_state.caps_results = caps_results
    st.session_state.caps_character = character_name

    # 파일로 저장도 1회만!
    if save_to_file:
        with open(f'caps_results_{character_name}.txt', 'w', encoding='utf-8') as f:
            for sig in caps_results:
                f.write(str(sig) + '\n')

    # Streamlit 결과 표시
    if caps_results:
        for sig in caps_results:
            st.markdown(f"```\n{sig}\n```")
    else:
        st.warning("행동서명 분석 결과가 없습니다.")

    return caps_results

def extract_json_from_codeblock(text):
    """
    코드블록(```...```)이 있을 경우 그 안의 내용만 추출해서 반환.
    없으면 전체를 반환.
    """
    # ```로 감싼 블록이 있으면 그 안의 내용만 추출
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()

def safe_brace(text):
    return text.replace("{", "{{").replace("}", "}}")

# Streamlit 앱의 메인 함수
def main():
    st.title("HEXACO-PI-R 성격 설문 분석 및 소설 등장인물 챗봇")
    caps_results = []
    # 등장인물 리스트 버튼
    if 'selected_character' not in st.session_state:
        with st.spinner("등장인물 분석 중..."):
            character_list = get_characters_list()
            st.session_state['character_list'] = character_list
        st.header("1단계: 등장인물 선택")
        st.info("아래 등장인물 중 분석 및 대화를 원하는 인물을 선택하세요.")
        cols = st.columns(3)
        for idx, character in enumerate(st.session_state['character_list']):
            if cols[idx % 3].button(character):
                st.session_state['selected_character'] = character
                st.session_state.responses_submitted = False
                st.session_state.hexaco_analysis_done = False
                st.session_state.final_hexako_result = ""
                st.session_state.chat_history = []
                st.rerun()
        if 'selected_character' not in st.session_state:
            st.stop()

    character_name = st.session_state['selected_character']
    st.success(f"선택된 등장인물: **{character_name}**")

    # 분석
    if not st.session_state.responses_submitted:
        st.header(f"{character_name}의 HEXACO 성격 분석")
        st.write(f"아래 **'결과 보기'** 버튼을 클릭하면 **{character_name}**의 HEXACO 성격 설문 응답과 CAPS 행동서명 분석을 동시에 진행합니다.")

        user_responses = get_user_responses_for_character(character_name, HEXACO_QUESTIONS)
        if st.button("결과 보기"):
            with st.spinner(f"{character_name}의 성격 및 CAPS 행동서명 분석 중입니다..."):
                # ----------- HEXACO 분석 ----------
                main_factor_scores, facet_scores, altruism_score = calculate_hexaco_scores(
                    user_responses, HEXACO_SCORING_KEY_FACETS, HEXACO_FACTORS_ORDER, ALTRUISM_KEY
                )
                st.session_state.main_factor_scores = main_factor_scores
                st.session_state.facet_scores = facet_scores
                st.session_state.altruism_score = altruism_score
                st.session_state.responses_submitted = True
                st.session_state.hexaco_analysis_done = True

                # HEXACO 결과 저장
                final_hexako_parts = [f"--- {character_name}의 HEXACO 성격 분석 결과 ---"]
                for factor in HEXACO_FACTORS_ORDER:
                    main_score = main_factor_scores.get(factor)
                    if pd.isna(main_score):
                        final_hexako_parts.append(f"** {factor}: 응답 부족으로 계산 불가 **")
                    else:
                        main_interpretation = interpret_score(main_score, factor)
                        final_hexako_parts.append(f"** {factor}: {main_score:.2f} - {main_interpretation}")
                        for facet_name, facet_score in facet_scores.get(factor, {}).items():
                            if pd.isna(facet_score):
                                final_hexako_parts.append(f" - {facet_name}: 응답 부족으로 계산 불가")
                            else:
                                facet_interpretation = interpret_score(facet_score, facet_name)
                                final_hexako_parts.append(
                                    f" - {facet_name}: {facet_score:.2f} - {facet_interpretation}")
                if not pd.isna(altruism_score):
                    altruism_interpretation = interpret_score(altruism_score, 'Altruism')
                    final_hexako_parts.append(
                        f"** Altruism (interstitial facet scale): {altruism_score:.2f} - {altruism_interpretation}")
                final_hexako_parts.append("--------------------------------------")
                st.session_state.final_hexako_result = "\n".join(final_hexako_parts)
                system_prompt_content = f"""
                            너는 사용자가 올린 소설 속 이야기를 기반으로 대화하는 등장인물이다. 너는 소설 속 '{character_name}'처럼 말하고 행동해야 해.
                            네 성격은 다음 Hexaco 결과에 기반을 두고 있어:
                            Hexaco 결과: {st.session_state.final_hexako_result}

                            그리고 너의 행동 경향성은 심리학의 CAPS 이론(Cognitive-Affective Personality System)과 DIAMONDS 이론을 활용해 추출된 행동 서명 결과를 참고해야 해.
                            아래는 DIAMONDS 분석 및 행동 서명 예시야:

                            {chr(10).join(f'- {safe_brace(sig)}' for sig in caps_results)}

                            **너의 임무:**
                            위의 성격 특성과 행동 경향성을 반드시 참고하여, 사용자의 입력(상황 및 대화)에 대해 가장 '{character_name}'답게, 그리고 일관성 있게 반응해야 해.
                            상황의 맥락과 내면의 가치관, 반복적 행동패턴(행동 서명)을 깊이 반영해서 대화하라.
                        """
                prompt2 = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt_content),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                    ]
                )
                runnable2 = prompt2 | model1
                with_message_history2 = RunnableWithMessageHistory(
                    runnable2,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                )

                # 세션에 저장 (챗봇 진입 전 이미 준비됨!)
                st.session_state.prompt2 = prompt2
                st.session_state.runnable2 = runnable2
                st.session_state.with_message_history2 = with_message_history2

                # **챗봇 LLM 세션 미리 초기화까지**
                st.session_state.with_message_history2.invoke(
                    {"input": HumanMessage(content=f"""{novelFullText}... 여기까지가 소설 내용이야. 넌 여기서 나오는 등장인물이고
                                넌 이 소설 내용 속 등장인물 '{character_name}'로서, 독자인 '나'와 이 소설에 대해 자유롭게 이야기해 줘. 등장인물의 성격과 감정을 담아서 말이야. 예를 들어, 소설 속 어떤 부분이 가장 인상 깊었는지, 특정 인물이나 사건에 대해 어떻게 생각하는지 등, 등장인물 관점에서 설명해 줘.
                                이제 나와 대화해 줘. """)},
                    config={"configurable": {"session_id": f"abcd_{character_name}_chat"}},
                )
                st.session_state.system_prompt_init = True

            st.rerun()

    # 결과 및 챗봇
    if st.session_state.responses_submitted:
        st.write("---")
        st.subheader(f"HEXACO 성격 점수 결과 - {character_name}")
        main_factor_scores = st.session_state.main_factor_scores
        facet_scores = st.session_state.facet_scores
        altruism_score = st.session_state.altruism_score

        # 레이더 차트 그리기
        st.header(f"{character_name}의 HEXACO 6가지 주요 요인 레이더 차트")
        radar_categories = [factor for factor in HEXACO_FACTORS_ORDER if not pd.isna(main_factor_scores.get(factor))]
        radar_scores = [main_factor_scores[factor] for factor in radar_categories]
        if radar_scores:
            fig = create_radar_chart(radar_scores, radar_categories, f"{character_name}의 HEXACO 주요 요인")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("레이더 차트를 그릴 충분한 데이터가 없습니다.")

        st.subheader(f"{character_name}의 HEXACO 점수 (평균) 및 해석")
        for factor in HEXACO_FACTORS_ORDER:
            main_score = main_factor_scores.get(factor)
            if pd.isna(main_score):
                st.write(f"** {factor}: 응답 부족으로 계산 불가 **")
            else:
                main_interpretation = interpret_score(main_score, factor)
                st.write(f"** {factor}: {main_score:.2f} (최소 1.00, 최대 5.00) - {main_interpretation}")
            for facet_name, facet_score in facet_scores.get(factor, {}).items():
                if pd.isna(facet_score):
                    st.write(f" - {facet_name}: 응답 부족으로 계산 불가")
                else:
                    facet_interpretation = interpret_score(facet_score, facet_name)
                    st.write(f" - {facet_name}: {facet_score:.2f} - {facet_interpretation}")

        if not pd.isna(altruism_score):
            altruism_interpretation = interpret_score(altruism_score, 'Altruism')
            st.write(
                f"\n** Altruism (interstitial facet scale): {altruism_score:.2f} (최소 1.00, 최대 5.00) - {altruism_interpretation}")

        st.subheader("표준 집단 (대학생 표본)과의 비교")
        st.write("(자기보고 기준, Z-점수는 표준 집단 대비 상대적 위치를 나타냄)")

        # 주요 요인 Z-점수 비교
        st.markdown("### 주요 HEXACO 요인 Z-점수")
        for factor in HEXACO_FACTORS_ORDER:
            user_score = main_factor_scores.get(factor)
            if pd.isna(user_score) or factor not in STANDARD_MEANS_SELF_REPORT:
                continue
            mean_std = STANDARD_MEANS_SELF_REPORT[factor]
            sd_std = STANDARD_SDS_SELF_REPORT[factor]
            z_score = (user_score - mean_std) / sd_std
            st.write(f"{factor}: 내 점수 {user_score:.2f} (표준 평균 {mean_std:.2f}, 표준편차 {sd_std:.2f}, Z-점수 {z_score:.2f})")

        # 하위 척도 Z-점수 비교
        st.markdown("### HEXACO 하위 척도 Z-점수")
        for factor in HEXACO_FACTORS_ORDER:
            st.markdown(f"** {factor}의 하위 척도:")
            for facet_name, user_facet_score in facet_scores.get(factor, {}).items():
                if pd.isna(user_facet_score) or facet_name not in STANDARD_MEANS_SELF_REPORT_FACETS.get(factor, {}):
                    st.write(f" - {facet_name}: 데이터 부족 또는 표준 데이터 없음")
                    continue
                mean_std = STANDARD_MEANS_SELF_REPORT_FACETS[factor][facet_name]
                sd_std = STANDARD_SDS_SELF_REPORT_FACETS[factor][facet_name]
                z_score = (user_facet_score - mean_std) / sd_std
                st.write(
                    f" - {facet_name}: 내 점수 {user_facet_score:.2f} (표준 평균 {mean_std:.2f}, 표준편차 {sd_std:.2f}, Z-점수 {z_score:.2f})")

        if not pd.isna(altruism_score) and 'Altruism' in STANDARD_MEANS_SELF_REPORT_FACETS:
            mean_std = STANDARD_MEANS_SELF_REPORT_FACETS['Altruism']
            sd_std = STANDARD_SDS_SELF_REPORT_FACETS['Altruism']
            z_score = (altruism_score - mean_std) / sd_std
            st.write(
                f"\n** Altruism: 내 점수 {altruism_score:.2f} (표준 평균 {mean_std:.2f}, 표준편차 {sd_std:.2f}, Z-점수 {z_score:.2f})")

        st.markdown("### Z-점수 해석 가이드")
        st.info("""
    * **Z-점수 0 근처**: 표준 집단(대학생)의 평균과 유사
    * **Z-점수 양수 (+)**: 표준 집단 평균보다 해당 특성이 높음
    * **Z-점수 음수 (-)**: 표준 집단 평균보다 해당 특성이 낮음
    * (예: Z-점수 +1.0은 약 상위 16%, -1.0은 약 하위 16% 수준)
    """)
        st.write("이 결과는 개인의 성격 특성이 특정 집단(이 경우 대학생 표본)과 비교했을 때")
        st.write("어느 정도 수준인지 상대적으로 이해하는 데 도움이 됩니다.")
        st.warning("참고: 하위 척도는 내적 일관성 신뢰도가 높지 않을 수 있으며, 주요 요인 점수가 더 신뢰할 수 있습니다.")

        st.write("---")
        st.header(f"{character_name}의 소설 내 상황-행동 패턴 분석 (CAPS)")

        if 'caps_results' not in st.session_state or st.session_state.get('caps_character') != character_name:
                analyze_caps_signatures(
                    character_name,
                    text_chunks,
                    llm_metadata,
                    extract_json_from_codeblock,
                    st,
                    save_to_file=True
                )

        else:
            # 이미 분석 결과가 있으면 재사용 (LLM 호출 없이 바로 출력)
            for sig in st.session_state.caps_results:
                st.markdown(f"```\n{sig}\n```")
        # CAPS 결과는 항상 표시
        if "caps_results" in st.session_state and st.session_state.caps_results:
            st.header(f"{character_name}의 소설 내 상황-행동 패턴 분석 (CAPS)")
            for sig in st.session_state.caps_results:
                st.markdown(f"```\n{sig}\n```")


        st.write("---")
        st.header(f"2단계: {character_name}와 대화하기")
        st.write(f"이제 {character_name}의 성격 분석 결과를 토대로 {character_name}와 대화할 수 있습니다.")

        # 채팅 메시지 표시
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 채팅 입력 처리
        # 채팅 입력 처리 (이제 if "system_prompt_init" 체크 안 해도 됨!)
        if prompt := st.chat_input(f"{character_name}에게 말을 걸어보세요!"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(f"{character_name}가 생각 중..."):
                    # 바로 응답 (이미 초기화되어 있음)
                    response_content = st.session_state.with_message_history2.invoke(
                        {"input": HumanMessage(content=prompt)},
                        config={"configurable": {"session_id": f"abcd_{character_name}_chat"}},
                    )
                    if response_content is not None:
                        full_response = response_content.content
                        st.markdown(full_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})


if __name__ == '__main__':
    main()