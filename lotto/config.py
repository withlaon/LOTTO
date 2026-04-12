import os
from dotenv import load_dotenv

load_dotenv()


def get_supabase_url() -> str:
    url = os.getenv("SUPABASE_URL", "").strip()
    if not url:
        raise RuntimeError("환경 변수 SUPABASE_URL을 설정하세요.")
    return url.rstrip("/")


def get_supabase_key() -> str:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv(
        "SUPABASE_ANON_KEY", ""
    ).strip()
    if not key:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY 또는 SUPABASE_ANON_KEY를 설정하세요."
        )
    return key
