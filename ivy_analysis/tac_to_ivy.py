import ast
from itertools import combinations

import tac
import tac_analysis

ivy_type_decls = """\
#lang ivy1.3


type python_object
type python_type


relation type_error
init ~type_error
conjecture ~type_error

individual type_of(X:python_object) : python_type


individual t_intbool : python_type
individual t_str : python_type
individual t_none : python_type

axiom t_intbool ~= t_str
axiom t_intbool ~= t_none
axiom t_str ~= t_none
axiom T = t_intbool | T = t_str | T = t_none


action addition(t1:python_type,t2:python_type) returns (t:python_type) = {
    if t1 = t_intbool & t2 = t_intbool {
        t := t_intbool
    } else {
        if (t1=t_str & t2=t_str) {
                t := t_str
        } else {
            type_error := true
        }
    }
}


action multiply(t1:python_type,t2:python_type) returns (t:python_type) = {
    if t1 = t_intbool & t2 = t_intbool {
        t := t_intbool
    } else {
        if ((t1 = t_str & t2 = t_intbool) | (t2 = t_str & t1 = t_intbool)) {
            t := t_str
        }
        else {
            type_error := true
        }
    }
}


action numeric_op(t1:python_type,t2:python_type) returns (t:python_type) = {
    if t1 = t_intbool & t2 = t_intbool {
        t := t_intbool
    } else {
        type_error := true
    }
}


action logical_not(t1:python_type) returns (t:python_type) = {
    t := t_intbool
}

action equal(t1:python_type, t2:python_type) returns (t:python_type) = {
    t := t_intbool
}

action comparison(t1:python_type,t2:python_type) returns (t:python_type) = {
    t := t_intbool;
    if t1 ~= t2 {
        type_error := true
    } else {
        if t1 = t_none {
            type_error := true
        }
    }
}


"""

def test_function():
    x = 0
    y = 1 < True
    while True:
        z = x + y
        x = "a"
        #y = "b"
    if (x > 0):
        y = 1
    else:
        y = 2
    z = x + y


# def test_function():
#     x = 1
#     y = '' #None
#     if (x > 0):
#         y = 1
#     else:
#         y = 2
#     z = x + y

# def test_function():
#     x1 = a + b
#     x2 = a ** b
#     x3 = x1 * x2
#     y = not x1
#     z1 = 1
#     z2 = '1'
#     z3 = True
#     z4 = False
#     z5 = None
#     # if (ha>3):
#     #    print('yes')
#     # z6 = -z1

#     # if x:
#     #     y = 1
#     # else:
#     #     x = 1


t_intbool = 't_intbool'
t_str = 't_str'
t_none = 't_none'


def parse_literal_expression(string_expr):
    if tac.is_stackvar(string_expr):
        return ast.Name
    body = ast.parse(string_expr).body
    assert len(body) == 1
    expr = body[0]
    return type(expr.value)

def ivy_and(*exprs):
    if len(exprs) == 0:
        return "true"
    else:
        return '(' + ' & '.join(
            '({})'.format(e) for e in exprs
        ) + ')'

def ivy_or(*exprs):
    if len(exprs) == 0:
        return "false"
    else:
        return '(' + ' | '.join(
            '({})'.format(e) for e in exprs
        ) + ')'



class TacToIvy(object):

    def ivy_type_of(self, x):
        x = x.strip()
        ast_class = parse_literal_expression(x)
        if ast_class is ast.Name:
            name = str(x)
            # get rid of '@'
            name = name.replace('@', 'tmp_')
            self.python_vars.add(name)
            return 'type_of({})'.format(name)
        elif ast_class is ast.Num:
            return t_intbool
        elif ast_class is ast.Str:
            return t_str
        elif ast_class is ast.NameConstant and x in ['True', 'False']:
            return t_intbool
        elif ast_class is ast.NameConstant and x == 'None':
            return t_none
        else:
            assert False, (x, ast_class)

    def instruction_to_ivy(self, instruction):
        # return instruction.fmt.format(**instruction._asdict()) + '\n'
        op = instruction.opcode
        if op == tac.OP.NOP:
            result = self.no_change()
        elif op == tac.OP.ASSIGN:
            result = self.assign(instruction)
        elif op == tac.OP.IMPORT:
            result = self.no_change()
        elif op == tac.OP.BINARY:
            result = self.binary(instruction)
        elif op == tac.OP.INPLACE:
            assert False
        elif op == tac.OP.CALL:
            result = self.call(instruction)
        elif op == tac.OP.JUMP:
            result = self.no_change()
        elif op == tac.OP.FOR:
            assert False
        elif op == tac.OP.RET:
            result = self.no_change()
            # assert False, "Function calls not supported"
        elif op == tac.OP.DEL:
            result = self.no_change()
        else:
            assert False, "Unknown Three Address Code instruction in {}".format(instruction)

        return result


    def no_change(self):
        return ""

    def assign(self, instruction):
        lhs = instruction.gens[0]
        rhs = instruction.uses[0]
        return "{} := {}".format(self.ivy_type_of(lhs), self.ivy_type_of(rhs))

    def call(self, instruction):
        assert len(instruction.gens) == 1
        return self.builtin_function(
            result=instruction.gens[0],
            function_name=instruction.func,
            arguments=instruction.uses[1:],  # TODO: add kargs (see TAC)
        )

    def binary(self, instruction):
        assert len(instruction.gens) == 1
        assert len(instruction.uses) == 2
        return self.builtin_function(
            result=instruction.gens[0],
            function_name=instruction.op,
            arguments=instruction.uses[:],
        )

    def builtin_function(self, result, function_name, arguments):
        function_name = function_name.strip()

        if function_name in ['**', '//', '%', '-', '<<', '>>', '&', '|', '^']:
            assert len(arguments) == 2
            return "{} := numeric_op({}, {})".format(
                self.ivy_type_of(result), self.ivy_type_of(arguments[0]), self.ivy_type_of(arguments[1])
            )

        elif function_name == '+':
            assert len(arguments) == 2
            return "{} := addition({}, {})".format(
                self.ivy_type_of(result), self.ivy_type_of(arguments[0]), self.ivy_type_of(arguments[1])
            )

        elif function_name == '*':
            assert len(arguments) == 2
            return "{} := multiply({}, {})".format(
                self.ivy_type_of(result), self.ivy_type_of(arguments[0]), self.ivy_type_of(arguments[1])
            )

        elif function_name in ['>', '<', '>=', '<=']:
            assert len(arguments) == 2
            return "{} := comparison({}, {})".format(
                self.ivy_type_of(result), self.ivy_type_of(arguments[0]), self.ivy_type_of(arguments[1])
            )

        elif function_name == 'not':
            assert len(arguments) == 1
            return "{} := logical_not({})".format(
                self.ivy_type_of(result), self.ivy_type_of(arguments[0])
            )

        elif function_name == '==':
            assert len(arguments) == 2
            return "{} := equal({}, {})".format(
                self.ivy_type_of(result), self.ivy_type_of(arguments[0]), self.ivy_type_of(arguments[1])
            )

        else:
            assert False, "builtin_function({!r}, {!r}, {!r})".format(
                result, function_name, arguments
            )

    def block_instructions_to_ivy(self, block_instructions):
        instructions = [self.instruction_to_ivy(instruction) for instruction in block_instructions]
        instructions = [x for x in instructions if x != '']
        return instructions

    def tac_blocks_cfg_to_ivy(self, tac_blocks_cfg):
        self.python_vars = set()
        block_ids = sorted(tac_blocks_cfg.nodes())
        first_block_id = block_ids[0]  # the smallest id is the first to run
        ivy_can_run_decls = "\n".join([
            "type basic_block",
            "relation can_run(B:basic_block)",
        ] + [
            "individual bb{}:basic_block".format(i)
            for i in block_ids
        ] + [
            "axiom bb{} ~= bb{}".format(i, j)
            for i, j in combinations(block_ids, 2)
        ] + [
            "axiom " + ivy_or(*("B = bb{}".format(i) for i in block_ids)),
            "init can_run(B) <-> B = bb{}".format(first_block_id)
        ]) + "\n\n"

        actions = []
        for i in block_ids:
            block_instructions = tac_blocks_cfg.node[i][tac.BLOCKNAME]
            successor_ids = tac_blocks_cfg.successors(i)
            ivy_instructions = (
                ["assume can_run(bb{})".format(i)] +
                self.block_instructions_to_ivy(block_instructions) +
                ["can_run(B) := {}".format(ivy_or(*(
                    "B = bb{}".format(j) for j in successor_ids
                )))]
            )

            actions.append(
                "action basic_block_{} = {{\n    ".format(i) +
                ";\n    ".join(ivy_instructions) +
                "\n}}\nexport basic_block_{}".format(i)
            )

        python_vars = sorted(self.python_vars)
        ivy_python_vars_decls = "\n".join([
            "individual {}:python_object".format(i)
            for i in python_vars
        ] + [
            "axiom {} ~= {}".format(i, j)
            for i, j in combinations(python_vars, 2)
        ]
        )

        return "\n\n\n".join(
            [
                ivy_type_decls,
                ivy_can_run_decls,
                ivy_python_vars_decls,
            ] +
            actions
        )


def test(code_to_analyze):
    from pdb import set_trace
    # set_trace()
    #tac_blocks_cfg = tac_analysis.make_tacblock_cfg(code_to_analyze)
    tac_blocks_cfg = tac.make_tacblock_cfg(code_to_analyze)
    tac.print_3addr(tac_blocks_cfg)
    print()
    tac_to_ivy = TacToIvy()
    result = tac_to_ivy.tac_blocks_cfg_to_ivy(tac_blocks_cfg)
    open("ivy_file.ivy", 'w').write(result)
    print(result)

if __name__ == "__main__":
    test(test_function)
