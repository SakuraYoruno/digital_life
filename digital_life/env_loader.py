"""Windows 环境变量补全。

Windows 上已经运行的进程（终端、IDE、宿主应用）不会自动继承后来修改的
用户/系统环境变量；CLI 启动时把注册表里的 DIGITAL_LIFE_* 补进 os.environ，
避免必须重启进程。这个模块不依赖任何宿主。
"""

from __future__ import annotations

import os
import sys

ENV_KEYS = (
    "DIGITAL_LIFE_API_KEY",
    "DIGITAL_LIFE_BASE_URL",
    "DIGITAL_LIFE_BIG_MODEL",
    "DIGITAL_LIFE_BIG_API_KEY",
    "DIGITAL_LIFE_SMALL_MODEL",
    "DIGITAL_LIFE_SMALL_BASE_URL",
    "DIGITAL_LIFE_SMALL_API_KEY",
    "DIGITAL_LIFE_SMALL_REASONING_EFFORT",
    "DIGITAL_LIFE_BIG_REASONING_EFFORT",
    "DIGITAL_LIFE_EMBED_MODEL",
    "DIGITAL_LIFE_EMBED_DIM",
    "DIGITAL_LIFE_EMBED_KEEP_ALIVE",
    "DIGITAL_LIFE_OLLAMA_URL",
    "DIGITAL_LIFE_DATA_DIR",
    "DIGITAL_LIFE_VAULT",
    "DIGITAL_LIFE_MOCK",
)

_USER_ENV = r"Environment"
_MACHINE_ENV = (
    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
)


def load_windows_environment(keys: tuple[str, ...] = ENV_KEYS) -> list[str]:
    """把注册表中缺失的变量补进当前进程；返回实际补入的变量名。"""
    if os.name != "nt" or sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:  # pragma: no cover
        return []

    loaded: list[str] = []
    locations = (
        (winreg.HKEY_CURRENT_USER, _USER_ENV),
        (winreg.HKEY_LOCAL_MACHINE, _MACHINE_ENV),
    )
    for hive, path in locations:
        try:
            with winreg.OpenKey(hive, path) as key:
                for name in keys:
                    if os.environ.get(name):
                        continue
                    try:
                        value, _kind = winreg.QueryValueEx(key, name)
                    except OSError:
                        continue
                    if value:
                        os.environ[name] = str(value)
                        loaded.append(name)
        except OSError:
            continue
    return loaded


__all__ = ["ENV_KEYS", "load_windows_environment"]
