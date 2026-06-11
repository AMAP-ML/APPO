import ast
import logging
import subprocess
from typing import Tuple

from verl.workers.agent.tools.base_tool import BaseTool

logger = logging.getLogger(__file__)


class PythonTool(BaseTool):
    """Python code execution tool using local conda environment subprocess."""

    def __init__(self, conda_path: str, conda_env: str):
        self.conda_path = conda_path
        self.conda_env = conda_env
        self.python_path = f"{conda_path}/envs/{conda_env}/bin/python"

    @property
    def name(self) -> str:
        return "python_interpreter"

    @property
    def trigger_tag(self) -> str:
        return "python"

    def execute(self, code: str, timeout: int = 120, **kwargs) -> str:
        """Execute Python code and return result."""
        result, report = self._run_code(code, timeout)
        return result if report == "Done" else report

    def _run_code(self, code: str, timeout: int) -> Tuple[str, str]:
        """Run Python code in conda environment and return result and status."""
        code = self._preprocess_code(code)
        try:
            process = subprocess.run(
                [self.python_path, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if process.returncode == 0:
                return process.stdout.strip(), "Done"
            else:
                logger.warning("[PythonTool] returncode=%d, stderr='%s'", process.returncode, process.stderr.strip())
                return "", process.stderr.strip()
        except subprocess.TimeoutExpired:
            return "", f"Execution timeout (exceeded {timeout}s)"
        except Exception as exc:
            return "", f"Execution exception: {exc}"

    def _preprocess_code(self, code: str) -> str:
        """Convert a bare trailing expression into a print() call."""
        try:
            tree = ast.parse(code)
            if tree.body:
                last_node = tree.body[-1]
                if isinstance(last_node, ast.Expr):
                    is_print_call = (
                        isinstance(last_node.value, ast.Call)
                        and isinstance(last_node.value.func, ast.Name)
                        and last_node.value.func.id == "print"
                    )
                    if not is_print_call:
                        tree.body[-1] = ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id="print", ctx=ast.Load()),
                                args=[last_node.value],
                                keywords=[],
                            )
                        )
                        code = ast.unparse(tree)
        except Exception:
            pass
        return code