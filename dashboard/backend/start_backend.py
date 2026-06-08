#!/usr/bin/env python
"""Backend entrypoint – applies cuda patch (no-op when CUDA is available).

Copyright 2025-2026 Fujitsu Ltd.
"""

import sys

import cpu_patch  # noqa: F401
import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload="--reload" in sys.argv,
    )
