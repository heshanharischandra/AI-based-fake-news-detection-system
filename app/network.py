"""Use the operating system's trusted certificates for model downloads.

Initialize before importing HTTP/model libraries. Hostname and certificate
verification remain enabled; custom CA environment settings are not replaced.
"""
from threading import Lock

_lock = Lock()
_configured = False


def configure_tls():
    """Idempotently enable native certificate validation for this process."""
    global _configured
    with _lock:
        if _configured:
            return
        import truststore

        truststore.inject_into_ssl()
        _configured = True
