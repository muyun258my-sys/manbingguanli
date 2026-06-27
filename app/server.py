from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple

from .models import ChatRequest
from .services import Orchestrator


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _parse_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


class Xm2RequestHandler(BaseHTTPRequestHandler):
    orchestrator = Orchestrator()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, 200, self.orchestrator.health())
            return
        if self.path.startswith("/profile/"):
            user_id = self.path.rsplit("/", 1)[-1]
            _json_response(self, 200, self.orchestrator.get_profile(user_id))
            return
        _json_response(self, 404, {"code": 404, "message": "not found", "data": None, "disclaimer": ""})

    def do_PUT(self) -> None:  # noqa: N802
        if not self.path.startswith("/profile/"):
            _json_response(self, 404, {"code": 404, "message": "not found", "data": None, "disclaimer": ""})
            return

        user_id = self.path.rsplit("/", 1)[-1]
        payload = _parse_json(self)
        result = self.orchestrator.update_profile(
            user_id,
            conditions=payload.get("conditions"),
            medications=payload.get("medications"),
            allergies=payload.get("allergies"),
        )
        _json_response(self, 200, result)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/chat":
            _json_response(self, 404, {"code": 404, "message": "not found", "data": None, "disclaimer": ""})
            return

        payload = _parse_json(self)
        missing = [key for key in ("session_id", "user_id", "message") if not payload.get(key)]
        if missing:
            _json_response(
                self,
                400,
                {
                    "code": 400,
                    "message": f"missing required field(s): {', '.join(missing)}",
                    "data": None,
                    "disclaimer": "",
                },
            )
            return

        request = ChatRequest(
            session_id=str(payload["session_id"]),
            user_id=str(payload["user_id"]),
            message=str(payload["message"]),
        )
        _json_response(self, 200, self.orchestrator.chat(request))


def run(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), Xm2RequestHandler)
    print(f"app listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
