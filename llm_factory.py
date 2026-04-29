from config import (
    PROVIDER,

    # Upstage
    UPSTAGE_API_KEY,
    UPSTAGE_BASE_URL,
    UPSTAGE_LLM_MODEL,
    UPSTAGE_EMBED_MODEL,

    # OpenAI
    OPENAI_API_KEY,
    OPENAI_LLM_MODEL,
    OPENAI_EMBED_MODEL,

    # Gemma
    GEMMA_API_KEY,
    GEMMA_BASE_URL,
    GEMMA_LLM_MODEL,
    GEMMA_EMBED_MODEL,
)


def get_llm():
    """LangChain LLM — structured output, chain용"""

    if PROVIDER == "upstage":
        from langchain_upstage import ChatUpstage

        return ChatUpstage(
            api_key=UPSTAGE_API_KEY,
            base_url=UPSTAGE_BASE_URL
        )

    elif PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_LLM_MODEL
        )

    elif PROVIDER == "gemma":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=GEMMA_API_KEY,
            base_url=GEMMA_BASE_URL,
            model=GEMMA_LLM_MODEL
        )

    else:
        raise ValueError(
            f"지원하지 않는 PROVIDER: {PROVIDER}"
        )


def get_embedding():
    """임베딩 모델"""

    if PROVIDER == "upstage":
        from langchain_upstage import UpstageEmbeddings

        return UpstageEmbeddings(
            upstage_api_key=UPSTAGE_API_KEY,
            model=UPSTAGE_EMBED_MODEL
        )

    elif PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model=OPENAI_EMBED_MODEL
        )

    elif PROVIDER == "gemma":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=GEMMA_API_KEY,
            base_url=GEMMA_BASE_URL,
            model=GEMMA_EMBED_MODEL
        )

    else:
        raise ValueError(
            f"지원하지 않는 PROVIDER: {PROVIDER}"
        )


def get_client():
    """openai.OpenAI 클라이언트 — chat.completions.create 직접 호출용"""

    from openai import OpenAI

    if PROVIDER == "upstage":
        return OpenAI(
            api_key=UPSTAGE_API_KEY,
            base_url=UPSTAGE_BASE_URL
        )

    elif PROVIDER == "openai":
        return OpenAI(
            api_key=OPENAI_API_KEY
        )

    elif PROVIDER == "gemma":
        return OpenAI(
            api_key=GEMMA_API_KEY,
            base_url=GEMMA_BASE_URL
        )

    else:
        raise ValueError(
            f"지원하지 않는 PROVIDER: {PROVIDER}"
        )