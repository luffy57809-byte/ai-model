"""
Per-user rate limiting via slowapi. Uses a custom key function that
decodes the JWT directly from the request header - NOT per-IP, since
per-IP would unfairly throttle multiple legitimate users behind the
same NAT/VPN/office network, while per-user is precise and fair given
every endpoint already requires login.

Falls back to a shared "anonymous" bucket only for requests without a
valid token - shouldn't normally happen since guarded endpoints reject
those with 401 before ever reaching a real handler, but this keeps the
rate limiter itself from crashing on a malformed/missing header.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.auth.auth import _decode_token


def _rate_limit_key(request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ")
        try:
            return f"user:{_decode_token(token)}"
        except Exception:
            pass
    return f"anon:{get_remote_address(request)}"


limiter = Limiter(key_func=_rate_limit_key)
