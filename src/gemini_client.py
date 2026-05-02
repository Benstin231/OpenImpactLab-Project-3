import os
import time
import threading
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from dotenv import load_dotenv

load_dotenv()

_client = None
_client_lock = threading.Lock()

DEFAULT_MODEL = "gemma-4-31b-it"
MAX_RETRIES = 3
RETRY_DELAY = 3


def get_client() -> genai.Client:
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY not set in environment or .env file")
            _client = genai.Client(api_key=api_key)
    return _client


def get_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def generate(contents: str, config: types.GenerateContentConfig) -> genai.types.GenerateContentResponse:
    """Call generate_content with automatic retry on 5xx errors."""
    client = get_client()
    model = get_model()
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except ServerError as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    raise last_exc
