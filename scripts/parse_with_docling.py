#!/data/data/com.termux/files/usr/bin/python3
import json
import sys
from pathlib import Path


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def fail(message: str, *, details: str | None = None, exit_code: int = 1) -> None:
    payload = {"ok": False, "error": message}
    if details:
        payload["details"] = details
    emit(payload)
    raise SystemExit(exit_code)


def main() -> None:
    if len(sys.argv) < 2:
        fail("Usage: parse_with_docling.py <document-path>")

    source = Path(sys.argv[1]).expanduser().resolve()
    if not source.is_file():
        fail(f"Document not found: {source}")

    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # pragma: no cover - environment dependent
        fail(
            "Docling is not installed for python3. Install it with: python3 -m pip install --user docling",
            details=str(exc),
        )

    try:
        converter = DocumentConverter()
        result = converter.convert(str(source))
        markdown = result.document.export_to_markdown()
    except Exception as exc:  # pragma: no cover - runtime/doc dependent
        fail(f"Docling failed to parse {source.name}: {exc}")

    content = str(markdown or "").replace("\x00", "").strip()
    if not content:
        fail(f"Docling returned no text for {source.name}")

    emit({
        "ok": True,
        "engine": "docling",
        "content": content,
    })


if __name__ == "__main__":
    main()
