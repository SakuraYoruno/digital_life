"""CLI 默认值：各子命令的 --data-dir 必须互相独立。

踩过的坑：给 sim 子命令 set_defaults(data_dir="data/sim") 会把 im/chat 的默认目录
一起改掉——argparse 的 parents 是按引用共享 action 的。机器人因此读写了一晚上的
仿真目录，还把仿真里的假用户当成真人去搭话。
"""

from __future__ import annotations

from digital_life.__main__ import _default_data_dir, build_parser


def test_data_dir_default_is_shared_across_subcommands():
    parser = build_parser()
    expected = str(_default_data_dir())
    for argv in (["chat"], ["im"], ["reflect"], ["sim"], ["demo"]):
        assert parser.parse_args(argv).data_dir == expected, argv


def test_sim_switches_dir_inside_its_command():
    """sim 的 data/sim 只在 cmd_sim 内部生效，不能污染解析层。"""
    import inspect

    from digital_life import __main__ as cli

    source = inspect.getsource(cli.cmd_sim)
    assert "data/sim" in source or "DIGITAL_LIFE_SIM_DIR" in source
