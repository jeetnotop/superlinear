import pytest


def test_parse_tool_call_block_basic():
    from superlinear.engine.tool_parser import parse_tool_call_block

    block = """<tool_call>
<function=my_tool>
<parameter=q>
hello
</parameter>
<parameter=n>3</parameter>
</function>
</tool_call>"""

    parsed = parse_tool_call_block(block)
    assert parsed.name == "my_tool"
    assert parsed.arguments["q"] == "hello"
    assert parsed.arguments["n"] == "3"


def test_parse_tool_call_block_json_value():
    from superlinear.engine.tool_parser import parse_tool_call_block

    block = """<tool_call>
<function=tool>
<parameter=args>
{"a": 1, "b": [2, 3]}
</parameter>
</function>
</tool_call>"""

    parsed = parse_tool_call_block(block)
    assert parsed.arguments["args"] == {"a": 1, "b": [2, 3]}


def test_parse_tool_call_block_multiline_preserved():
    from superlinear.engine.tool_parser import parse_tool_call_block

    block = """<tool_call>
<function=tool>
<parameter=text>
line1
line2
</parameter>
</function>
</tool_call>"""

    parsed = parse_tool_call_block(block)
    assert parsed.arguments["text"] == "line1\nline2"


def test_parse_tool_call_block_missing_wrapper_raises():
    from superlinear.engine.tool_parser import ToolCallParseError, parse_tool_call_block

    with pytest.raises(ToolCallParseError):
        parse_tool_call_block("<function=a></function>")

