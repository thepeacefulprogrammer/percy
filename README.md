# Pi Web Chat

Local mobile-friendly chat UI for Pi running in Termux.

## Start

```bash
cd ~/pi-web-chat
./start.sh
```

Open (default port 8787, override with `PORT`):

- http://127.0.0.1:8787
- or http://localhost:8787

## Stop

```bash
cd ~/pi-web-chat
./stop.sh
```

## Restart

```bash
cd ~/pi-web-chat
./restart.sh
```

## Status

```bash
cd ~/pi-web-chat
./status.sh
```

## Local test models

This repo includes helper scripts for two local llama.cpp-backed test models.
They now default to a `32768` token context window so they can handle Pi Web Chat's large system prompt/tool list:

### Existing RP test model

```bash
cd ~/pi-web-chat
./start-local-rp-model.sh
./status-local-rp-model.sh
./stop-local-rp-model.sh
```

### Qwen2.5 Coder 0.5B test model

```bash
cd ~/pi-web-chat
./start-local-qwen-coder-model.sh
./status-local-qwen-coder-model.sh
./stop-local-qwen-coder-model.sh
```

When the Qwen helper is running, Pi exposes it through `~/.pi/agent/models.json` as:

- provider: `local-qwen`
- model id: `qwen2.5-coder-0.5b-local`
- label in the web UI: `Local Qwen2.5 Coder 0.5B (Q4_K_M)`

## Notes

- This uses a dedicated Pi RPC session named `Web Chat`.
- Session files are stored in `~/.local/share/pi-web-chat/sessions`.
- It reuses your installed Pi config/model credentials.
- The server now runs under `supervisor.sh`, so if `server.js` crashes, it is automatically restarted.
- `runtime-env.sh` centralizes the shared shell runtime config (port, public URL, readiness URL, run directory).
- `start.sh`, `status.sh`, and the watchdog now respect `PORT`; the watchdog readiness probe defaults to `/api/readyz`.
- `server.js` now exposes `/api/livez` for process liveness and `/api/readyz` (also `/api/healthz`) for RPC readiness.
- Binary document attachments such as PDF, DOCX, PPTX, XLSX, EPUB, RTF, and OpenDocument files can be parsed through Docling and injected into the prompt as extracted Markdown.
- Docling is optional and currently invoked through `scripts/parse_with_docling.py`. Install it for `python3` with:

```bash
python3 -m pip install --user docling
```
