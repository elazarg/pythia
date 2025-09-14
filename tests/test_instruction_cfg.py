import pytest
import dis
from dis import Instruction
import spyte.graph_utils as gu
from spyte.instruction_cfg import (
    is_unconditional_jump,
    is_return,
    is_raise,
    is_sequencer,
    is_jump_source,
    is_jump,
    next_list,
    calculate_stack_effect,
    calculate_stack_depth,
    pos_str,
    make_instruction_block_cfg,
)


# Test functions to analyze
def simple_function():
    x = 1
    y = 2
    return x + y


def function_with_jumps(x):
    if x > 0:
        return x
    else:
        return -x


def function_with_loop(n):
    sum = 0
    for i in range(n):
        sum += i
    return sum


def test_next_list():
    # Create instructions for testing
    bytecode = dis.Bytecode(function_with_jumps)
    instructions = list(bytecode)

    # Test next_list for different types of instructions
    for i, ins in enumerate(instructions[:-1]):
        fallthrough = instructions[i + 1].offset
        next_instructions = next_list(ins, fallthrough)

        # Check that next_list returns a list of (label, stack_effect) pairs
        assert isinstance(next_instructions, list)
        for label, effect in next_instructions:
            assert label is None or isinstance(label, (int, float))
            assert isinstance(effect, int)

        # Check that sequencer instructions do not have fallthrough
        if is_sequencer(ins):
            assert not any(label == fallthrough for label, _ in next_instructions)

        # Check that jump instructions have their target in the next list
        if is_jump(ins):
            assert any(label == ins.argval for label, _ in next_instructions)


def test_calculate_stack_effect():
    # Create instructions for testing
    bytecode = dis.Bytecode(simple_function)
    instructions = list(bytecode)

    # Test calculate_stack_effect for different instructions
    for ins in instructions:
        # Test with jump=True
        effect_jump = calculate_stack_effect(ins, jump=True)
        assert isinstance(effect_jump, int)

        # Test with jump=False
        effect_no_jump = calculate_stack_effect(ins, jump=False)
        assert isinstance(effect_no_jump, int)


def test_pos_str():
    # Create an instruction with positions
    bytecode = dis.Bytecode(simple_function)
    instructions = list(bytecode)

    # Test pos_str for instructions with positions
    for ins in instructions:
        if hasattr(ins, "positions") and ins.positions:
            pos_string = pos_str(ins.positions)
            assert isinstance(pos_string, str)
            assert ":" in pos_string  # Should contain line:col format


def test_make_instruction_block_cfg():
    # Test make_instruction_block_cfg for different functions
    for func in [simple_function, function_with_jumps, function_with_loop]:
        depths, cfg = make_instruction_block_cfg(func)

        # Check that depths is a dictionary mapping labels to stack depths
        assert isinstance(depths, dict)
        for label, depth in depths.items():
            assert isinstance(label, (int, float))
            assert isinstance(depth, int)

        # Check that cfg is a Cfg object
        assert isinstance(cfg, gu.Cfg)

        # Check that cfg contains blocks
        assert len(cfg.labels) > 0

        # Check that each block contains instructions
        for label in cfg.labels:
            if label != gu.EXIT_LABEL:  # Skip exit label
                block = cfg[label]
                assert len(block) > 0
                for ins in block:
                    assert isinstance(ins, Instruction)


def test_calculate_stack_depth():
    # Create a CFG for testing
    _, cfg = make_instruction_block_cfg(simple_function)

    # Test calculate_stack_depth
    depths = calculate_stack_depth(cfg)

    # Check that depths is a dictionary mapping labels to stack depths
    assert isinstance(depths, dict)
    for label, depth in depths.items():
        assert isinstance(label, (int, float))
        assert isinstance(depth, int)

    # Check that all labels in the CFG have a stack depth
    for label in cfg.labels:
        if label != gu.EXIT_LABEL:  # Skip exit label
            assert label in depths
