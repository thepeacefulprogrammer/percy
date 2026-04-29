import tempfile
import unittest
from pathlib import Path

from textual.app import App, ComposeResult

from percy.ui.path_input import PathCompletionInput, extract_path_fragment, get_path_completions


class PathCompletionTests(unittest.TestCase):
    def test_extracts_unquoted_fragment(self) -> None:
        value = "Please open percy/ui/ap"
        cursor = len(value)

        fragment = extract_path_fragment(value, cursor)

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.text, "percy/ui/ap")
        self.assertIsNone(fragment.quote)

    def test_extracts_fragment_inside_quotes(self) -> None:
        value = 'Read "docs/my fi" next'
        cursor = value.index('" next')

        fragment = extract_path_fragment(value, cursor)

        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertEqual(fragment.text, "docs/my fi")
        self.assertEqual(fragment.quote, '"')

    def test_completes_unique_relative_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            (cwd / "README.md").write_text("hi")

            completion = get_path_completions("REA", cwd)

            self.assertIsNotNone(completion)
            assert completion is not None
            self.assertEqual(completion.replacements, ["README.md"])
            self.assertEqual(completion.common_prefix, "README.md")

    def test_hides_dotfiles_without_dot_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            (cwd / ".env").write_text("x=1")
            (cwd / "app.py").write_text("print('hi')")

            completion = get_path_completions("a", cwd)

            self.assertIsNotNone(completion)
            assert completion is not None
            self.assertEqual(completion.replacements, ["app.py"])

    def test_appends_slash_for_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            (cwd / "percy").mkdir()
            (cwd / "perimeter").mkdir()

            completion = get_path_completions("pe", cwd)

            self.assertIsNotNone(completion)
            assert completion is not None
            self.assertEqual(completion.replacements, ["percy/", "perimeter/"])
            self.assertEqual(completion.common_prefix, "per")

    def test_quotes_paths_with_spaces_when_not_already_quoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            (cwd / "my file.txt").write_text("hi")

            completion = get_path_completions("my", cwd)

            self.assertIsNotNone(completion)
            assert completion is not None
            self.assertEqual(completion.replacements, ['"my file.txt"'])

    def test_preserves_unquoted_completion_inside_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            (cwd / "my file.txt").write_text("hi")

            completion = get_path_completions("my", cwd, quote='"')

            self.assertIsNotNone(completion)
            assert completion is not None
            self.assertEqual(completion.replacements, ["my file.txt"])


class _SubmitCaptureApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield PathCompletionInput(id="prompt")

    async def on_input_submitted(self, event: PathCompletionInput.Submitted) -> None:
        self.submitted.append(event.value)


class PathCompletionInputSubmitTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_posts_input_submitted_event(self) -> None:
        app = _SubmitCaptureApp()

        async with app.run_test() as pilot:
            input_widget = app.query_one(PathCompletionInput)
            input_widget.value = "hello"

            await input_widget.action_submit()
            await pilot.pause()

            self.assertEqual(app.submitted, ["hello"])


if __name__ == "__main__":
    unittest.main()
