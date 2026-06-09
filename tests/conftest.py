"""
pytest 配置文件 — 确保测试隔离性。

关键：清理 DPO_TEST_* 环境变量，防止测试间状态泄漏。
"""

import os
import pytest


@pytest.fixture(autouse=True)
def clean_dpo_env():
    """
    每个测试前后自动清理 DPO 测试环境变量。

    防止 test_dpo_edit 设置的 DPO_TEST_ACTION=edit
    泄漏到后续测试中导致无限循环。
    """
    # 保存旧值
    old_values = {
        k: os.environ.get(k)
        for k in ("DPO_TEST_ACTION", "DPO_TEST_NEW_TIER", "DPO_TEST_REJECT_REASON")
    }

    # 清除
    for k in old_values:
        os.environ.pop(k, None)

    yield  # 测试执行

    # 恢复（或清除）
    for k, v in old_values.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
