"""Wire protocol between the simulator (Isaac Sim's Python) and the model servers (their own containers).

Both sides only need the standard library and numpy: requests and replies are pickled dicts sent over a
``multiprocessing.connection`` socket on localhost (all containers use the host network).

    SmolVLA server   act      {"images": [overhead, side, wrist], "task": str, "state": float32[8]}
                              -> {"actions": float32[T, 8]}
    Molmo2 server    inspect  {"images": [top, side]} -> {"defective": bool, "verdict": str, "reason": str}

Images are HxWx3 uint8 arrays. Every reply also carries "latency" (s), or "error" with a traceback.
"""

from __future__ import annotations

import os
import time
import traceback
from multiprocessing.connection import Client, Listener

POLICY_ADDRESS = ("127.0.0.1", int(os.environ.get("POLICY_PORT", "6123")))
VLM_ADDRESS = ("127.0.0.1", int(os.environ.get("VLM_PORT", "6124")))
AUTHKEY = b"pi-final-policy"


class ModelClient:
    def __init__(self, address=POLICY_ADDRESS):
        self.conn = Client(address, authkey=AUTHKEY)

    def call(self, cmd: str, **kwargs) -> dict:
        self.conn.send({"cmd": cmd, **kwargs})
        reply = self.conn.recv()
        if "error" in reply:
            raise RuntimeError(f"model server: {reply['error']}")
        return reply

    def act(self, images, task: str, state) -> dict:
        return self.call("act", images=list(images), task=task, state=state)

    def inspect(self, images) -> dict:
        return self.call("inspect", images=list(images))

    def close(self) -> None:
        self.conn.close()


def serve(handlers: dict, address, name: str) -> None:
    """Answer requests forever, one client at a time; ``handlers`` maps a command to fn(**request) -> dict."""
    with Listener(address, authkey=AUTHKEY) as listener:
        print(f"[{name}] listening on {address[0]}:{address[1]}", flush=True)
        while True:
            conn = listener.accept()
            print(f"[{name}] client connected", flush=True)
            while True:
                try:
                    req = conn.recv()
                except (EOFError, ConnectionResetError):
                    print(f"[{name}] client disconnected", flush=True)
                    break
                t0 = time.time()
                cmd = req.pop("cmd")
                try:
                    reply = {"ok": True} if cmd == "ping" else handlers[cmd](**req)
                except KeyError:
                    reply = {"error": f"unknown command {cmd!r}"}
                except Exception:  # keep serving; the client raises with the traceback
                    reply = {"error": traceback.format_exc()}
                reply["latency"] = time.time() - t0
                conn.send(reply)
