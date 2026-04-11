import subprocess
import sys
from pathlib import Path
from typing import Any

from agent_framework import Skill, SkillScript, SkillsProvider


def add_skills_root(skills_root_path: str | Path) -> None:
    if isinstance(skills_root_path, Path):
        skills_root_path = str(skills_root_path)
    if skills_root_path not in sys.path:
        sys.path.insert(0, skills_root_path)


def subprocess_script_runner(
    skill: Skill, script: SkillScript, args: dict[str, Any] | None = None
) -> str:
    if not skill.path:
        return f"Error: Skill '{skill.name}' has no directory path."

    if not script.path:
        return (
            f"Error: Script '{script.name}' has no file path. "
            "Only file-based scripts can be executed locally."
        )

    script_path = Path(skill.path) / script.path
    if not script_path.is_file():
        return f"Error: Script file not found: {script_path}"

    cmd = [sys.executable, str(script_path)]

    if args:
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_path.parent),
        )
    except subprocess.TimeoutExpired:
        return f"Error: Script '{script.name}' timed out after 30 seconds."
    except OSError as exc:
        return f"Error: Failed to execute script '{script.name}': {exc}"

    output = result.stdout
    if result.stderr:
        output += f"\nStderr:\n{result.stderr}"
    if result.returncode != 0:
        output += f"\nScript exited with code {result.returncode}"

    return output.strip() or "(no output)"


def get_skills_provider(skills_dir: Path) -> SkillsProvider | None:
    if not skills_dir.exists():
        return None

    add_skills_root(skills_dir)
    return SkillsProvider(
        skill_paths=str(skills_dir),
        script_runner=subprocess_script_runner,
    )
