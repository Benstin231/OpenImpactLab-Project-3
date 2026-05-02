import os
import threading
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = None
_client_lock = threading.Lock()

DEFAULT_MODEL = "gemma-4-31b-it"


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
