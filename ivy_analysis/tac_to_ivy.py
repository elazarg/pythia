import ast

import tac
import tac_analysis


def test_function():
    x1 = a + b
    x2 = a ** b
    x3 = x1 * x2
    y = not x1
    z1 = 1
    z2 = '1'
    z3 = True
    z4 = False
    z5 = None
    # z6 = -z1

    # if x:
    #     y = 1
    # else:
    #     x = 1

def parse_literal_expression(string_expr):
    if tac.is_stackvar(string_expr):
        return ast.Name
    body = ast.parse(string_expr).body
    assert len(body) == 1
    expr = body[0]
    return type(expr.value)

t_int = 't_int'
t_str = 't_str'
t_bool = 't_bool'
t_none = 't_none'

def ivy_type_of(x):
    x = x.strip()
    ast_class = parse_literal_expression(x)
    if ast_class is ast.Name:
        return 'type({})'.format(x)
    elif ast_class is ast.Num:
        return t_int
    elif ast_class is ast.Str:
        return t_str
    elif ast_class is ast.NameConstant and x in ['True', 'False']:
        return t_bool
    elif ast_class is ast.NameConstant and x == 'None':
        return t_none
    else:
        assert False, (x, ast_class)

class TacToIvy(object):
    def instruction_to_ivy(self, instruction):
        # return instruction.fmt.format(**instruction._asdict()) + '\n'
        op = instruction.opcode
        if op == tac.OP.NOP:
            return self.no_change()
        elif op == tac.OP.ASSIGN:
            return self.assign(instruction)
        elif op == tac.OP.IMPORT:
            return self.no_change()
        elif op == tac.OP.BINARY:
            return self.binary(instruction)
        elif op == tac.OP.INPLACE:
            assert False
        elif op == tac.OP.CALL:
            return self.call(instruction)
        elif op == tac.OP.JUMP:
            return self.no_change()
        elif op == tac.OP.FOR:
            assert False
        elif op == tac.OP.RET:
            return self.no_change()
            assert False, "Function calls not supported"
        elif op == tac.OP.DEL:
            return self.no_change()
        else:
            assert False, "Unknown Three Address Code instruction in {}".format(instruction)

    def no_change(self):
        return ""

    def assign(self, instruction):
        lhs = instruction.gens[0]
        rhs = instruction.uses[0]

        return """
        type({}) := {}
        """.format(lhs, ivy_type_of(rhs))

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

    def ivy_and(self, *exprs):
        return '(' + ' & '.join(
            '({})'.format(e) for e in exprs
        ) + ')'

    def builtin_function(self, result, function_name, arguments):
        function_name = function_name.strip()

        # numeric only operations
        if function_name in ['**', '//', '/', '%', '-', '<<', '>>']:
            return """
            if {} {{
                type({}) := t_int
            }} else {{
                type_error := true
            }}
            """.format(self.ivy_and(*('type({})=t_int'.format(x) for x in arguments)), result)

        elif function_name == '+':
            return """
            if ((type({left})=t_int | type({left})=t_bool) & (type({right})=t_int | type({right})=t_bool)) {{
                type({result}) := t_int
            }} else {{
                if (type({left})=t_str & type({right})=t_str) {{
                    type({result}) := t_str
                }} else {{
                    type_error := true
                }}
            }}
            """.format(result=result, left=arguments[0], right=arguments[1])

        elif function_name == '*':
            return """
            if (type({left})=t_int & type({right})=t_int) {{
                type({result}) := t_int
            }} else {{
                if ((type({left})=t_str & type({right})=t_int) | (type({right})=t_str & type({left})=t_int)) {{
                    type({result}) := t_str
                }} else {{
                    type_error := true
                }}
            }}
            """.format(result=result, left=arguments[0], right=arguments[1])


        # if function_name in ['&', '|', '^']:
        #     # TODO: not support implicit conversion to boolean
        #     return all_boolean_implies_boolean

        elif function_name == 'not':
            return """
            type({result}) := t_bool
            """.format(result=result)

        else:
            assert False, "builtin_function({!r}, {!r}, {!r})".format(result, function_name, arguments)

    def tac_blocks_cfg_to_ivy(self, tac_blocks_cfg):
        st = """
        relation str(X)
        relation num(X)
        relation bool(X)
        function type : object -> type
        
        """
        first_block_id = min(tac_blocks_cfg.nodes())
        for node_id in tac_blocks_cfg.nodes():
            block_instructions = tac_blocks_cfg.node[node_id][tac.BLOCKNAME]
            successor_ids = tac_blocks_cfg.successors(node_id)


            st += "action basic_block_{} = {{\n".format(node_id)
            for instruction in block_instructions:
                st += self.instruction_to_ivy(instruction)
            st += "}\n"
        return st

def test(code_to_analyze):
    tac_blocks_cfg = tac_analysis.make_tacblock_cfg(code_to_analyze)
    tac_analysis.print_tac_cfg(tac_blocks_cfg)
    print()
    tac_to_ivy = TacToIvy()
    print(tac_to_ivy.tac_blocks_cfg_to_ivy(tac_blocks_cfg))

if __name__ == "__main__":
    test(test_function)
