# Pi Web Chat

Local mobile-friendly chat UI for Pi running in Termux.

## Start

```bash
cd ~/pi-web-chat
./start.sh
```

Open:

- http://127.0.0.1:8787
- or http://localhost:8787

## Stop

```bash
cd ~/pi-web-chat
./stop.sh
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
