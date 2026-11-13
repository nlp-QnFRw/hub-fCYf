"""统一的问答模板：预训练数据、SFT 数据、推理 prompt 共用同一份。

模板只有一处定义，避免「预训练一种格式、SFT 另一种格式」造成的分布错位::

    问：{instruction} 答：{output}<eos>

约定：

* 一条记录 = 一行 = 一段问答，行尾**恰好一个** ``<eos>``；
* ``<eos>`` 是唯一终止符：SFT 生成时遇到它即停；
* 推理时用 :func:`format_prompt` 得到 ``问：{instruction} 答：``，模型接着续写答案；
* SFT 训练时建议只对 ``答：`` 之后的部分计算 loss（prompt 部分 mask 掉），
  但 prompt 必须与这里完全一致。

用法::

    from prompt import QA_TEMPLATE, format_prompt, format_qa

    text = format_qa("1+1 等于几？", "等于 2。", add_eos=True)
    # '问：1+1 等于几？ 答：等于 2。<eos>'

    prompt = format_prompt("1+1 等于几？")
    # '问：1+1 等于几？ 答：'
"""

from __future__ import annotations

from tokenizer import EOS_TOKEN

#: 问句前缀
Q_PREFIX = "问："
#: 答案标记（前置空格，把问句和答案分开）
A_MARK = " 答："
#: 完整模板：``QA_TEMPLATE.format(instruction=..., output=...)``
QA_TEMPLATE = Q_PREFIX + "{instruction}" + A_MARK + "{output}"

__all__ = ["Q_PREFIX", "A_MARK", "QA_TEMPLATE", "format_prompt", "format_qa"]


def format_prompt(instruction: str) -> str:
    """推理用的输入前缀：``问：{instruction} 答：``（不含答案与 ``<eos>``）。"""
    return Q_PREFIX + instruction + A_MARK


def format_qa(instruction: str, output: str, add_eos: bool = False) -> str:
    """把一条问答拼成训练文本。

    Args:
        instruction: 问题 / 指令。
        output: 答案。
        add_eos: 为真时在末尾追加终止符 ``<eos>``（训练数据用）。
    """
    text = QA_TEMPLATE.format(instruction=instruction, output=output)
    return text + EOS_TOKEN if add_eos else text
