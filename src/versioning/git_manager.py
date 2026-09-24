import subprocess
from typing import Dict, Any
import os

class GitManager:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        
    def _run_git_command(self, cmd: list) -> str:
        try:
            result = subprocess.run(
                ["git"] + cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def get_current_version_info(self) -> Dict[str, str]:
        """Extracts the current git repository version details."""
        repo_name = os.path.basename(os.path.abspath(self.repo_path))
        
        branch = self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        commit_hash = self._run_git_command(["rev-parse", "HEAD"])
        commit_date = self._run_git_command(["log", "-1", "--format=%cd", "--date=iso"])
        
        return {
            "repository": repo_name,
            "branch": branch if branch else "unknown",
            "commit_hash": commit_hash if commit_hash else "unknown",
            "commit_date": commit_date if commit_date else "unknown"
        }
