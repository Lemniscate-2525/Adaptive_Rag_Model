
"""Production security configuration."""

import os
import secrets
from urllib.parse import urlparse


def validate_security_settings():
    production = os.getenv("APP_ENV", "development").lower() == "production"

    jwt_secret = os.getenv("JWT_SECRET_KEY", "")

    if len(jwt_secret.encode("utf-8")) < 32:
        raise RuntimeError("JWT_SECRET_KEY must contain at least 32 bytes.")

    if production:
        if jwt_secret.lower() in {
            "secret",
            "changeme",
            "your-secret-key",
        }:
            raise RuntimeError("Insecure JWT secret.")

        if os.getenv("ENFORCE_HTTPS", "false").lower() != "true":
            raise RuntimeError("HTTPS enforcement must be enabled in production.")

        origins = os.getenv("CORS_ORIGINS", "")
        if not origins.strip():
            raise RuntimeError("CORS_ORIGINS must be configured.")

        for origin in origins.split(","):
            parsed = urlparse(origin.strip())
            if parsed.scheme != "https" or not parsed.netloc:
                raise RuntimeError("Production CORS origins must use HTTPS.")

        hosts = os.getenv("ALLOWED_HOSTS", "")
        if not hosts.strip():
            raise RuntimeError("ALLOWED_HOSTS must be configured.")

    return production
