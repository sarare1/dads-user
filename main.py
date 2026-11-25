import os
import webbrowser
from threading import Timer

import uvicorn

HOST = os.environ.get("DADS_HOST", "127.0.0.1")   # this machine only; 0.0.0.0 exposes it to the network
PORT = int(os.environ.get("DADS_PORT", "8000"))
RELOAD = os.environ.get("DADS_RELOAD") == "1"      # auto-restart on code changes (development only)


def open_browser():
    webbrowser.open_new(f"http://localhost:{PORT}")


if __name__ == "__main__":
    print("=" * 70)
    print("  DADS - Open-Set Radar Emitter Recognition (local run)")
    print(f"  http://localhost:{PORT}   |   stop with Ctrl+C")
    print("=" * 70)

    if os.environ.get("DADS_NO_BROWSER") != "1":
        Timer(1.5, open_browser).start()

    uvicorn.run("src.api.server:app", host=HOST, port=PORT, reload=RELOAD)
