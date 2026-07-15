import os


DEAD_LOCAL_PROXIES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
    "http://localhost:9",
    "https://localhost:9",
}


def remove_dead_local_proxy() -> list[str]:
    """Elimina solo el proxy local nulo que bloquea conexiones externas."""
    removed = []
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        value = os.environ.get(name, "").strip().lower().rstrip("/")
        if value in DEAD_LOCAL_PROXIES:
            os.environ.pop(name, None)
            removed.append(name)
    return removed
