import os
import re
from collections.abc import Container, Mapping

from loguru import logger

from toolmaker.utils.paths import LOCAL_TOOLMAKER_DIR

ENV_VAR_SUBSTITUTION_PATTERN = re.compile(
    r"(?P<full>\$\{env:(?P<var>[A-Za-z_][A-Za-z0-9_]*)\})"
)
ALLOWED_ENV_VARS: Container[str] = {"HF_TOKEN"}


def substitute_env_vars(
    s: str,
    env: Mapping[str, str] | None = None,
    allowed: Container[str] = ALLOWED_ENV_VARS,
) -> str:
    if env is None:
        env = os.environ

    def substitute(match: re.Match) -> str:
        var = match.group("var")
        if var not in env:
            logger.warning(
                f"Unable to perform environment variable substitution for {var}: not found in environment"
            )
            return match.group("full")
        if var not in allowed:
            logger.warning(
                f"Unable to perform environment variable substitution for {var}: not in list of allowed environment variable substitutions ({allowed!r})"
            )
            return match.group("full")
        return env[var]

    return ENV_VAR_SUBSTITUTION_PATTERN.sub(substitute, s)


def get_env_dict_in_container() -> Mapping[str, str]:
    # Load environment variables
    env = dict(os.environ)
    # Add the toolmaker directory to the PATH, so that the subprocess_utils module can be imported
    env["PATH"] = env.get("PATH", "") + f":{LOCAL_TOOLMAKER_DIR!s}"
    return env
