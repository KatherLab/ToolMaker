from collections.abc import Mapping

from toolmaker.utils.env import substitute_env_vars


class _AllowAll:
    def __contains__(self, item: str) -> bool:
        return True


def test_no_substitution() -> None:
    """Test string with no environment variables."""
    assert substitute_env_vars("hello world", allowed=_AllowAll()) == "hello world"


def test_single_env_var() -> None:
    """Test substitution of a single environment variable."""
    test_env = {"HOME": "/home/user"}
    assert (
        substitute_env_vars("${env:HOME}/docs", env=test_env, allowed=_AllowAll())
        == "/home/user/docs"
    )


def test_multiple_env_vars() -> None:
    """Test substitution of multiple environment variables."""
    test_env = {"USER": "john", "HOME": "/home/john"}
    assert (
        substitute_env_vars(
            "${env:USER}:${env:HOME}", env=test_env, allowed=_AllowAll()
        )
        == "john:/home/john"
    )


def test_missing_env_var() -> None:
    """Test behavior when environment variable is not found."""
    test_env: Mapping[str, str] = {}
    # Should keep the original ${env:NONEXISTENT} when variable is not found
    assert (
        substitute_env_vars("${env:NONEXISTENT}", env=test_env, allowed=_AllowAll())
        == "${env:NONEXISTENT}"
    )


def test_nested_env_vars() -> None:
    """Test that nested-looking variables are processed correctly."""
    test_env = {"FOO": "bar", "BAR": "baz"}
    # Should process one variable at a time
    assert (
        substitute_env_vars("${env:FOO}${env:BAR}", env=test_env, allowed=_AllowAll())
        == "barbaz"
    )


def test_empty_env_var() -> None:
    """Test substitution with empty environment variable value."""
    test_env = {"EMPTY": ""}
    assert (
        substitute_env_vars(
            "prefix${env:EMPTY}suffix", env=test_env, allowed=_AllowAll()
        )
        == "prefixsuffix"
    )


def test_none_env() -> None:
    """Test that function works with system environment when env=None."""
    import os

    os.environ["TEST_VAR"] = "test_value"
    try:
        assert (
            substitute_env_vars("${env:TEST_VAR}", allowed=_AllowAll()) == "test_value"
        )
    finally:
        del os.environ["TEST_VAR"]
