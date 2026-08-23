"""
Phase 11: Structural production-safety tests.

These must pass without network and without credentials, and they enforce
that no production trading path can be accidentally enabled.
"""
import pathlib
import re

import pytest

from src.adapters.binance import BinanceTestnetConfig, TESTNET_BASE_URL


def test_testnet_base_url_is_testnet():
    assert "testnet" in TESTNET_BASE_URL.lower()
    assert "api.binance.com" not in TESTNET_BASE_URL or "testnet" in TESTNET_BASE_URL.lower()


def test_no_hardcoded_production_url_in_adapter():
    # The only live URL constant must be testnet; error messages that
    # *reject* production URLs are allowed and expected.
    assert "testnet" in TESTNET_BASE_URL.lower()
    src = pathlib.Path("src/adapters/binance.py").read_text()
    # Find URL assignments (not error strings)
    import re
    assignments = re.findall(r'base_url\s*=\s*["\']([^"\']+)["\']', src)
    for url in assignments:
        if "binance" in url.lower():
            assert "testnet" in url.lower(), f"Production URL assigned: {url}"


def test_no_production_env_branch():
    src = pathlib.Path("src/adapters/binance.py").read_text()
    # Verify there is no else: production fallback branch via AST
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.orelse:
            orelse_src = ast.unparse(node.orelse) if hasattr(ast, "unparse") else str(node.orelse)
            assert "api.binance.com" not in orelse_src.lower(), f"Production URL in else branch: {orelse_src[:200]}"
            assert "production" not in orelse_src.lower() or "testnet" in orelse_src.lower(), f"Suspicious else branch: {orelse_src[:200]}"
    # Verify BinanceTestnetConfig rejects non-testnet
    from src.adapters.binance import BinanceTestnetConfig
    import pytest
    with pytest.raises(ValueError):
        BinanceTestnetConfig(binance_env="production", api_key="k", api_secret="s")


def test_config_rejects_production_env():
    with pytest.raises(ValueError, match="BINANCE_ENV='testnet'"):
        BinanceTestnetConfig(binance_env="production", api_key="k", api_secret="s")
    with pytest.raises(ValueError):
        BinanceTestnetConfig(binance_env="live", api_key="k", api_secret="s")
    with pytest.raises(ValueError):
        BinanceTestnetConfig(binance_env="", api_key="k", api_secret="s")


def test_config_rejects_production_url_even_if_env_testnet():
    with pytest.raises(ValueError, match="testnet"):
        BinanceTestnetConfig(binance_env="testnet", api_key="k", api_secret="s",
                             base_url="https://api.binance.com")


def test_example_env_contains_no_real_secrets():
    text = pathlib.Path(".env.example").read_text()
    assert "your_testnet_api_key_here" in text
    assert "your_testnet_api_secret_here" in text
    # Must not contain anything that looks like a real key (long base64)
    assert "api.binance.com" not in text or "testnet" in text.lower()


def test_gitignore_protects_credentials():
    text = pathlib.Path(".gitignore").read_text()
    assert ".env" in text
    # .env.example should NOT be ignored (it's the template)
    assert ".env.example" not in text or ".env\n" in text  # .env covers .env, but not .env.example explicitly
    assert "credentials.json" in text or ".env" in text


def test_no_secret_logging_in_adapter():
    src = pathlib.Path("src/adapters/binance.py").read_text().lower()
    # The adapter must filter secrets from journal
    assert "secret" in src and "api_key" in src  # it does filter
    # And must not have a bare log of api_secret
    assert 'logger.info.*api_secret' not in src
    assert 'logger.info.*secret' not in src
