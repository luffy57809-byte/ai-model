"""
JWT-based authentication: password hashing (bcrypt via passlib),
JWT creation/verification, and a get_current_user FastAPI dependency
that guards protected endpoints.

SECRET_KEY comes from the JWT_SECRET_KEY env var - MUST be set to a
real random secret in production (Render deployment). Falls back to a
fixed dev-only value locally so tests/local dev don't require extra
setup, but this fallback is NEVER safe to use in production (anyone
could forge tokens if the secret is guessable/known).
"""

import datetime
import os
import uuid

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Header

from src.storage.database import SessionLocal, User

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-only-insecure-secret-do-not-use-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24 * 7  # 1 week

# Using bcrypt directly rather than passlib - passlib 1.7.4 (last
# released 2020, effectively unmaintained) hard-crashes against modern
# bcrypt versions (5.x removed an internal attribute passlib's
# version-detection code depended on). bcrypt's own API is simple
# enough that the abstraction layer wasn't adding real value anyway.
BCRYPT_MAX_BYTES = 72  # bcrypt's real, hard limit - truncate rather than crash


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")[:BCRYPT_MAX_BYTES]
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    password_bytes = password.encode("utf-8")[:BCRYPT_MAX_BYTES]
    return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))


def create_access_token(user_id: str) -> str:
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired - please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")


def get_current_user_id(authorization: str = Header(None)) -> str:
    """FastAPI dependency - add `user_id: str = Depends(get_current_user_id)`
    to any endpoint that should require login. Raises 401 if the
    Authorization header is missing or the token is invalid/expired."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header - expected 'Bearer <token>'.",
        )
    token = authorization.removeprefix("Bearer ")
    return _decode_token(token)


def create_user(email: str, password: str) -> dict:
    session = SessionLocal()
    try:
        existing = session.query(User).filter(User.email == email).first()
        if existing:
            raise ValueError("An account with this email already exists.")

        user_id = uuid.uuid4().hex
        user = User(
            id=user_id, email=email, hashed_password=hash_password(password),
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        session.add(user)
        session.commit()
        return {"user_id": user_id, "email": email}
    finally:
        session.close()


def authenticate_user(email: str, password: str) -> str:
    """Returns the user_id if credentials are valid, raises ValueError otherwise."""
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user is None or not verify_password(password, user.hashed_password):
            raise ValueError("Incorrect email or password.")
        return user.id
    finally:
        session.close()
