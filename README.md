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
