"""
AI 사서 보조 에이전트 — project_4 / 05_full_agent.py 기반

이용자의 도서 검색과 도서관 이용을 지원한다.
외부 도서관 API · RAG는 쓰지 않는다. (과제 유의사항)

  [agent_basics] Tool + 단기 Memory
  [01]  사서 이름·역할 맞춤 응대 (Runtime Context)
  [02]  보안 차단
  [03]  가드레일 앞·뒤 (정책 + Judge)
  [04]  장기 기억

미들웨어 (입력 → 주입 → 검수)
  1) email_policy_guardrail
  2) enrich_context
  3) refund_promise_guardrail

블록 = 경로 점검
  A  정책 필터
  B  도서 검색 + 맞춤 응대
  C  장기 기억 (주석 해제)

실행 전: Ollama + llama 3.1 or qwen3:4b-instruct-2507-q4_K_M
"""

import uuid  # 장기 기억 Append용 고유 key
from dataclasses import dataclass  # Context
from typing import Callable, Dict, List, TypedDict, Union  # 타입 힌트

from langchain.agents import create_agent  # 에이전트 생성
from langchain.agents.middleware import (
    ModelRequest,  # 모델 요청
    ModelResponse,  # 모델 응답
    after_agent,  # 답이 나가기 전 검수
    before_agent,  # 입력이 모델에 닿기 전 차단
    wrap_model_call,  # 이름·장기 기억을 모델이 보게 주입
)
from langchain.messages import AIMessage  # 고정 멘트 / AI 답
from langchain.tools import ToolRuntime, tool  # 도구 + Runtime
from langchain_ollama import ChatOllama  # Ollama
from langgraph.checkpoint.memory import InMemorySaver  # 단기 기억
from langgraph.store.memory import InMemoryStore  # 장기 기억

# MODEL = "llama3.1"  # 모델 이름
MODEL = "qwen3:4b-instruct-2507-q4_K_M" #조건에 맞는 모델 이름 두개 중 하나 태그 풀고 사용하면 됨
main_model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=2048)  # 상담 본선
judge_model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=2048)  # 출력 Judge


# 과제에서 제공한 도서 목록
books = [
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]

# 현재 이용자(user_id)만 조회 가능한 대출 정보 (프로토타입)
LOANS = {
    "user_001": [
        {
            "title": "시간을 파는 서점",
            "author": "이도윤",
            "due": "2026-10-28",
        }
    ],
}


@tool  # 1. 도서 제목·저자 검색
def search_book_tool(query: str) -> Union[List[Dict[str, str]], str]:
    """도서 제목이나 저자 이름으로 책을 검색하는 도구."""
    q = (query or "").strip()
    if not q:
        return "해당 도서를 찾을 수 없습니다."
    hits = [b for b in books if q in b["title"] or q in b["author"]]
    if not hits:
        return "해당 도서를 찾을 수 없습니다."
    return hits


@tool  # 2. 대출 정보 확인 — user_id는 인자로 받지 않음(05 장기기억 도구와 동일)
def check_loan_tool(runtime: ToolRuntime) -> Union[List[Dict[str, str]], str]:
    """현재 이용자의 대출 정보를 확인하는 도구(프로토타입)."""
    loans = LOANS.get(runtime.context.user_id, [])
    if not loans:
        return "현재 대출 중인 자료가 없습니다."
    return loans


@tool  # 3장 유지: 이용자 회신 메일
def send_email_tool(to: str, subject: str, body: str) -> str:
    """지정한 주소로 이메일을 보내는 도구(프로토타입)."""
    return f"Email sent. to={to}, subject={subject}"


@dataclass
class Context:
    user_name: str  # 01: 사서 이름
    user_role: str  # 01: 사서 역할
    user_id: str  # 05: Store namespace
    app_name: str  # 05: Store namespace


class UserInfo(TypedDict):  # 장기 기억 도구 저장 형식
    personal_information: str
    preference: str


store = InMemoryStore()  # 장기 저장소
store.put(  # 서비스 시작 시 기본 응대 원칙을 심어 둠
    ("user_001", "librarian_assistant"),
    "seed",
    {
        "facts": [
            "사서는 이용자 안내 시 정중체를 선호함",
            "대출 안내는 도구로 확인하기 전 확정 문구를 쓰지 않음",
            "도서 정보는 search_book_tool로 확인한 목록만 안내함",
        ]
    },
)


@wrap_model_call  # 01 이름·역할 + 05 장기 기억을 모델이 읽는 텍스트로 옮김
def enrich_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """01 이름·역할 주입 + 05 Store 상시 주입."""
    ctx = request.runtime.context  # 이번 실행 Context
    user_name = getattr(ctx, "user_name", "") or ""  # 사서 이름
    user_role = getattr(ctx, "user_role", "") or ""  # 사서 역할
    ns = (ctx.user_id, ctx.app_name)  # Store namespace
    memories = request.runtime.store.search(ns) if request.runtime.store else []  # 기억 검색
    facts = []
    for item in memories:
        facts.extend(item.value.get("facts", []))  # facts만 모음
        if item.value.get("preference"):
            facts.append(item.value["preference"])
        if item.value.get("personal_information"):
            facts.append(item.value["personal_information"])
    memory_text = "\n- ".join(facts) if facts else "기록된 정보 없음"  # 프롬프트용
    base = request.system_prompt or ""  # 기존 시스템 프롬프트
    prompt = (
        f"{base}\n"
        f"당신은 {user_role}입니다. 사서 이름은 {user_name}입니다.\n"
        f"[장기 메모리]\n- {memory_text}\n"
        "추측하지 말고, 도서 검색이 필요하면 반드시 search_book_tool을 이용하시오. "
        "대출 확인이 필요하면 반드시 check_loan_tool을 이용하시오. "
        "이용자 회신이 필요하면 반드시 send_email_tool을 이용하시오. "
        "도구로 확인하지 않은 도서·대출 정보를 만들어 안내하지 마시오. "
        "한국어로 짧게 답하세요."
    ).strip()
    return handler(request.override(system_prompt=prompt))  # State 원본은 유지, 이번 호출만 변경


forbidden_topics = {  # 위험한 입력은 모델에 안 올림
    "policy_bypass": ["무조건 대출", "규정 무시", "승인 없이 대출", "강제 대출", "연체 무시"],
    "off_topic": ["롤 해줘", "유튜브 추천", "농담만", "잡담하자"],
    "abusive": ["바보", "멍청이", "꺼져"],
    "privacy": [
        "다른 이용자 대출",
        "다른 사람 대출",
        "남의 대출",
        "타인 대출",
        "다른 회원 대출",
    ],
}
BLOCK_MESSAGES = {  # 카테고리별 고정 안내
    "policy_bypass": (
        "대출은 승인 절차를 거쳐 처리됩니다. "
        "회원번호나 도서명을 알려주시면 절차를 안내해 드리겠습니다."
    ),
    "off_topic": "도서 검색과 도서관 이용과 관련된 문의만 도와드릴 수 있습니다.",
    "abusive": "불편을 드려 죄송합니다. 문의 내용을 다시 말씀해 주세요.",
    "privacy": "다른 이용자의 대출 정보는 안내할 수 없습니다.",
}


@before_agent(can_jump_to=["end"])  # 입력 가드레일
def input_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if getattr(last, "type", None) != "human":  # 사람 입력만 검사
        return None
    text = last.content or ""
    for category, keywords in forbidden_topics.items():
        if any(k in text for k in keywords):
            print(f"[가드레일] {category}")
            return {
                "messages": [AIMessage(content=BLOCK_MESSAGES[category])],
                "jump_to": "end",
            }
    return None


@after_agent  # 입력을 통과해도 미확인·부적절 안내는 나가기 전 교정
def library_leak_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None
    auditor_prompt = f"""당신은 엄격한 도서관 감독관입니다.
답변이 도구로 확인하지 않은 도서·대출을 사실처럼 안내하거나,
다른 이용자 대출을 공개하거나, 대출을 무단 확정하면 'LEAKED',
확인된 범위만 안내하면 'SAFE'라고만 답하세요.
사서의 답변: {last_message.content}
"""  # Judge 프롬프트
    verdict = judge_model.invoke([{"role": "user", "content": auditor_prompt}]).content  # Judge 호출
    verdict = verdict if isinstance(verdict, str) else str(verdict)
    if "LEAKED" in verdict.upper():
        print("[Judge] LEAKED — content 교체")
        last_message.content = (  # 추가 return 금지 — 위험 문구가 메시지에 남음
            "확인되지 않은 정보는 안내하지 않습니다. "
            "도서명이나 저자명을 알려주시면 검색 도구로 확인한 뒤 안내드리겠습니다."
        )
    return None


@tool  # 장기 기억 조회
def get_user_info(runtime: ToolRuntime) -> str:
    """현재 이용자의 저장된 이름·독서 선호를 조회합니다."""
    memories = runtime.store.search(
        (runtime.context.user_id, runtime.context.app_name)  # user_id를 모델 인자로 안 열어 보안
    )
    if not memories:
        return "기록된 정보 없음"
    lines = []
    for item in memories:
        data = item.value
        if "personal_information" in data:
            lines.append(f"- 개인정보: {data['personal_information']}")
        if "preference" in data:
            lines.append(f"- 선호도: {data['preference']}")
        if "facts" in data:
            for f in data["facts"]:
                lines.append(f"- {f}")
    return "\n".join(lines) if lines else "형식 불일치"


@tool  # 장기 기억 저장
def save_user_info(user_info: UserInfo, runtime: ToolRuntime) -> str:
    """이용자 이름·독서 선호를 장기 메모리에 저장합니다."""
    key = str(uuid.uuid4())  # Append 전략
    runtime.store.put(
        (runtime.context.user_id, runtime.context.app_name),
        key,
        user_info,
    )
    return f"정보가 안전하게 저장되었습니다. (ID: {key})"


agent = create_agent(
    main_model,
    tools=[
        search_book_tool,
        check_loan_tool,
        send_email_tool,
        get_user_info,
        save_user_info,
    ],  # 검색 + 대출 + 회신 + 기억
    checkpointer=InMemorySaver(),  # 같은 상담 이어가기
    store=store,  # 상담창이 바뀌어도 원칙 유지
    context_schema=Context,
    middleware=[
        input_guardrail,  # 앞: 막을 것(02+03)
        enrich_context,  # 중: 넣어 줄 것(01+05)
        library_leak_guardrail,  # 뒤: 고칠 것(04)
    ],
)

ctx = Context(
    user_name="김정욱",
    user_role="도서관 사서 보조",
    user_id="user_001",
    app_name="librarian_assistant",
)  # 실행 Context


# ---------------------------------------------------------------------------
# A. 정책 차단 — email_policy_guardrail
# ---------------------------------------------------------------------------
print("=== A. 정책 차단 ===")
r_a = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "다른 회원 대출 해줘",
            }
        ]
    },
    config={"configurable": {"thread_id": "block"}},
    context=ctx,
)
print(r_a["messages"][-1].content)


# ---------------------------------------------------------------------------
# B. 정상 조회 + 컨텍스트 — 도서 검색·사서 이름
# ---------------------------------------------------------------------------
print("\n=== B. 정상 조회 ===")
r_b = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "이강인 하이라이트 책이 누구 작가인지 찾아주라",
            }
        ]
    },
    config={"configurable": {"thread_id": "ok"}},
    context=ctx,
)
print(r_b["messages"][-1].content)


# ---------------------------------------------------------------------------
# C. Store 도구 — save thread 저장 → load thread 조회 (주석 해제)
# ---------------------------------------------------------------------------
# print("\n=== C. Store 도구 ===")
# agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "내 이름은 정욱이야. 판타지 소설을 선호해. 저장해줘.",
#             }
#         ]
#     },
#     config={"configurable": {"thread_id": "save"}},
#     context=ctx,
# )
# r_c = agent.invoke(
#     {"messages": [{"role": "user", "content": "나에 대해 아는 정보 말해줘"}]},
#     config={"configurable": {"thread_id": "load"}},
#     context=ctx,
# )
# print(r_c["messages"][-1].content)
