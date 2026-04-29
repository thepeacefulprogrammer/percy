from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from textual.binding import Binding
from textual.widgets import Input, OptionList


@dataclass(frozen=True)
class PathFragment:
    start: int
    end: int
    text: str
    quote: str | None = None


@dataclass(frozen=True)
class PathCompletion:
    replacements: list[str]
    common_prefix: str


@dataclass(frozen=True)
class CompletionSession:
    start: int
    end: int
    replacements: list[str]


def extract_path_fragment(value: str, cursor_position: int) -> PathFragment | None:
    """Extract the token around the cursor that should be completed as a path."""

    cursor_position = max(0, min(cursor_position, len(value)))

    active_quote: str | None = None
    quote_start = -1
    escaped = False
    for index, char in enumerate(value[:cursor_position]):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if active_quote is None:
            if char in {"'", '"'}:
                active_quote = char
                quote_start = index
        elif char == active_quote:
            active_quote = None
            quote_start = -1

    if active_quote is not None:
        start = quote_start + 1
        end = cursor_position
        escaped = False
        while end < len(value):
            char = value[end]
            if escaped:
                escaped = False
                end += 1
                continue
            if char == "\\":
                escaped = True
                end += 1
                continue
            if char == active_quote:
                break
            end += 1
        return PathFragment(
            start=start,
            end=end,
            text=value[start:end],
            quote=active_quote,
        )

    start = cursor_position
    while start > 0 and not value[start - 1].isspace():
        start -= 1

    end = cursor_position
    while end < len(value) and not value[end].isspace():
        end += 1

    if start == end:
        return None

    return PathFragment(start=start, end=end, text=value[start:end])


def get_path_completions(
    fragment_text: str, cwd: Path, *, quote: str | None = None
) -> PathCompletion | None:
    """Build path completion candidates for a fragment."""

    if not fragment_text:
        return None

    base_dir, name_prefix = _resolve_search_root(fragment_text, cwd)
    if not base_dir.exists() or not base_dir.is_dir():
        return None

    show_hidden = name_prefix.startswith(".")
    matches: list[Path] = []
    for entry in sorted(
        base_dir.iterdir(),
        key=lambda path: (not path.is_dir(), path.name.lower(), path.name),
    ):
        if not show_hidden and entry.name.startswith("."):
            continue
        if entry.name.startswith(name_prefix):
            matches.append(entry)

    if not matches:
        return None

    replacements = [
        _format_replacement(match, fragment_text, cwd, quote=quote) for match in matches
    ]
    return PathCompletion(
        replacements=replacements,
        common_prefix=os.path.commonprefix(replacements),
    )


class PathCompletionMenu(OptionList):
    DEFAULT_CSS = """
    PathCompletionMenu {
        display: none;
        height: auto;
        max-height: 8;
    }

    PathCompletionMenu.visible {
        display: block;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("markup", False)
        kwargs.setdefault("compact", True)
        super().__init__(*args, **kwargs)
        self.session: CompletionSession | None = None
        self.add_class("hidden")

    def show_completions(
        self, *, start: int, end: int, replacements: list[str]
    ) -> None:
        self.session = CompletionSession(start=start, end=end, replacements=replacements)
        self.set_options(replacements)
        self.highlighted = 0 if replacements else None
        self.remove_class("hidden")
        self.add_class("visible")

    def hide_completions(self) -> None:
        self.session = None
        self.clear_options()
        self.highlighted = None
        self.remove_class("visible")
        self.add_class("hidden")

    @property
    def visible_menu(self) -> bool:
        return self.session is not None and bool(self.options)

    def highlighted_replacement(self) -> str | None:
        session = self.session
        highlighted = self.highlighted
        if session is None or highlighted is None:
            return None
        if not (0 <= highlighted < len(session.replacements)):
            return None
        return session.replacements[highlighted]


class PathCompletionInput(Input):
    BINDINGS = [
        *Input.BINDINGS,
        Binding("tab", "complete_path", "Complete path", show=False),
        Binding("shift+tab", "complete_path_previous", show=False),
        Binding("down", "completion_cursor_down", show=False),
        Binding("up", "completion_cursor_up", show=False),
        Binding("escape", "hide_path_completions", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._suppress_menu_hide = False

    def _watch_value(self, value: str) -> None:
        super()._watch_value(value)
        if not self._suppress_menu_hide:
            self._hide_completion_menu()

    def action_complete_path(self) -> None:
        if self._accept_inline_suggestion():
            self._hide_completion_menu()
            return

        menu = self._completion_menu()
        if menu is not None and menu.visible_menu:
            menu.action_cursor_down()
            return

        fragment = extract_path_fragment(self.value, self.cursor_position)
        if fragment is None:
            self.app.bell()
            self._hide_completion_menu()
            return

        completion = get_path_completions(fragment.text, Path.cwd(), quote=fragment.quote)
        if completion is None or not completion.replacements:
            self.app.bell()
            self._hide_completion_menu()
            return

        if len(completion.replacements) == 1:
            self._replace_fragment(
                completion.replacements[0], fragment.start, fragment.end
            )
            return

        current_fragment_value = fragment.text
        if completion.common_prefix and completion.common_prefix != current_fragment_value:
            self._replace_fragment(
                completion.common_prefix, fragment.start, fragment.end
            )
            return

        if menu is None:
            self.app.bell()
            return

        menu.show_completions(
            start=fragment.start,
            end=fragment.end,
            replacements=completion.replacements,
        )

    def action_complete_path_previous(self) -> None:
        menu = self._completion_menu()
        if menu is not None and menu.visible_menu:
            menu.action_cursor_up()

    def action_completion_cursor_down(self) -> None:
        menu = self._completion_menu()
        if menu is not None and menu.visible_menu:
            menu.action_cursor_down()

    def action_completion_cursor_up(self) -> None:
        menu = self._completion_menu()
        if menu is not None and menu.visible_menu:
            menu.action_cursor_up()

    def action_hide_path_completions(self) -> None:
        self._hide_completion_menu()

    async def action_submit(self) -> None:
        if self._apply_highlighted_completion():
            return
        await super().action_submit()

    def _accept_inline_suggestion(self) -> bool:
        if self.cursor_at_end and self._suggestion:
            self.value = self._suggestion
            self.cursor_position = len(self.value)
            return True
        return False

    def _replace_fragment(self, replacement: str, start: int, end: int) -> None:
        self._suppress_menu_hide = True
        try:
            self.replace(replacement, start, end)
        finally:
            self._suppress_menu_hide = False
        self._hide_completion_menu()

    def _apply_highlighted_completion(self) -> bool:
        menu = self._completion_menu()
        if menu is None or not menu.visible_menu or menu.session is None:
            return False

        replacement = menu.highlighted_replacement()
        if replacement is None:
            return False

        self._replace_fragment(replacement, menu.session.start, menu.session.end)
        return True

    def _hide_completion_menu(self) -> None:
        menu = self._completion_menu()
        if menu is not None and menu.visible_menu:
            menu.hide_completions()

    def _completion_menu(self) -> PathCompletionMenu | None:
        try:
            return self.app.query_one("#path-completion-menu", PathCompletionMenu)
        except Exception:
            return None


def _resolve_search_root(fragment_text: str, cwd: Path) -> tuple[Path, str]:
    expanded = Path(fragment_text).expanduser()
    if not expanded.is_absolute():
        expanded = (cwd / expanded).resolve(strict=False)
    else:
        expanded = expanded.resolve(strict=False)

    if fragment_text.endswith("/") or fragment_text in {".", "..", "~", "/"}:
        return expanded, ""

    return expanded.parent, expanded.name


def _format_replacement(
    match: Path, fragment_text: str, cwd: Path, *, quote: str | None = None
) -> str:
    display = _display_path(match, fragment_text, cwd)
    if quote is None and any(char.isspace() for char in display):
        escaped = display.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return display


def _display_path(path: Path, fragment_text: str, cwd: Path) -> str:
    if fragment_text.startswith("~"):
        home = Path.home()
        try:
            relative = path.relative_to(home)
            display = f"~/{relative.as_posix()}" if relative.parts else "~"
        except ValueError:
            display = path.as_posix()
    elif fragment_text.startswith("/"):
        display = path.as_posix()
    else:
        display = os.path.relpath(path, cwd).replace(os.sep, "/")
        if fragment_text.startswith("./") and not display.startswith(("./", "../")):
            display = f"./{display}"

    if path.is_dir():
        display = f"{display}/"
    return display
