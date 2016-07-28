import ast
from itertools import combinations

import tac
import tac_analysis

ivy_type_decls = open('ivy_type_decls.ivy').read()

def test_function():
    #lst = ((1,2), ('a','b'))
    #lst = [(1,2), ('a', 'b',) , ([],[]), (((),),), 1+x]
    lst = []
    lst = lst + [(1,2)]
    lst = lst + [('a','b')]
    #lst.append((1,2))
    #lst.append(('a','b'))
    for x in lst:
        y = x[0] + x[1]
    #x,y = z
    #v0 = 1
    #v1 = 'a'
    #v2 = None
    #x = (v0, v1, v2) #1,'a',None)
    #y = x[2]

def test_function():
    # lst = []
    # lst = lst + [(1,2)]
    # lst = lst + [('a',None)]
    # for x in lst:
    #     i = x[0]
    #     j = x[1]
    #     k = i + 1

    lst = [('a', 'b')]
    lst = lst + [(1,2)]
    lst = lst + [([1],['a'])]
    #lst = lst + [([1],'a')]
    for x in lst:
        k = x[0] + x[1]

    # lst = []
    # lst = lst + [(1,2)]
    # lst = lst + [('a',None)]
    # for x in lst:
    #     y = x[0] + x[1]
    # z = 'a' + None

def test_function2():
    x = 0
    y = 1 < True
    while True:
        z = x + y
        x = "aaa"
        #y = "b"
        x = 2
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

    def ivy_type_of_lit(self, x):
        ast_class = parse_literal_expression(x)
        if ast_class is ast.Num:
            return 't_intbool'
        elif ast_class is ast.Str:
            return 't_str'
        elif ast_class is ast.NameConstant and x in ['True', 'False']:
            return 't_intbool'
        elif ast_class is ast.NameConstant and x == 'None':
            return 't_none'
        else:
            return None

    def ivy_obj_of(self, x):
        x = x.strip()
        ast_class = parse_literal_expression(x)
        if ast_class is ast.Name:
            if x not in self.python_vars:
                name = str(x)
                # get rid of '@'
                name = name.replace('@', 'tmp_')
                self.python_vars[x] = name
            return self.python_vars[x]
        elif self.ivy_type_of_lit(x) is not None:
            if x not in self.python_lits:
                self.python_lits[x] = 'lit_{}'.format(len(self.python_lits))
            return self.python_lits[x]
        else:
            assert False, (x, ast_class)

    def instruction_to_ivy(self, instruction):
        # return instruction.fmt.format(**instruction._asdict()) + '\n'
        # print(instruction)
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
            assert False, str(instruction)
        elif op == tac.OP.CALL:
            result = self.call(instruction)
        elif op == tac.OP.JUMP:
            result = self.no_change()
        elif op == tac.OP.FOR:
            result = self.for_loop(instruction)
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
        return "{} := {}".format(self.ivy_obj_of(lhs), self.ivy_obj_of(rhs))

    def for_loop(self, instruction):
        assert len(instruction.gens) == 1
        assert len(instruction.uses) == 1
        return self.ivy_func(instruction.gens[0], 'for_loop', instruction.uses)


    def call(self, instruction):
        assert len(instruction.gens) == 1
        result = instruction.gens[0]
        function_name = instruction.func.strip()
        arguments = instruction.uses[1:]  # TODO: add kargs (see TAC)

        if function_name in self.python_vars:
            assert False, str(instruction)
            # The following is not needed after Elazar's fix:
            #
            # assert len(arguments) <= 2, "call_X only up to X=2"
            # return "{} := call_{}({})".format(
            #     self.ivy_obj_of(result),
            #     len(arguments),
            #     ', '.join(self.ivy_obj_of(x) for x in [function_name] + list(arguments))
            # )
        else:
            return self.builtin_function(result, function_name, arguments)

    def binary(self, instruction):
        assert len(instruction.gens) == 1
        assert len(instruction.uses) == 2
        return self.builtin_function(
            result=instruction.gens[0],
            function_name=instruction.op,
            arguments=instruction.uses[:],
        )

    def ivy_func(self, result, ivy_function_name, arguments):
        return "tmp_result := {}{}\n{} := tmp_result".format(
            ivy_function_name,
            '(' + ', '.join(self.ivy_obj_of(a) for a in arguments) + ')' if len(arguments) > 0 else '',
            self.ivy_obj_of(result)
        )


    def builtin_function(self, result, function_name, arguments):
        function_name = function_name.strip()

        if function_name in ['**', '//', '%', '-', '<<', '>>', '&', '|', '^'] and len(arguments) == 2:
            return self.ivy_func(result, 'numeric_op', arguments)

        elif function_name == '+' and len(arguments) == 2:
            return self.ivy_func(result, 'addition', arguments)

        elif function_name == '+' and len(arguments) == 2:
            return self.ivy_func(result, 'multiply', arguments)

        elif function_name in ['>', '<', '>=', '<='] and len(arguments) == 2:
            return self.ivy_func(result, 'comparison', arguments)

        elif function_name == 'not' and len(arguments) == 1:
            return self.ivy_func(result, 'logical_not', arguments)

        elif function_name in ['>', '<', '>=', '<='] and len(arguments) == 2:
            return self.ivy_func(result, 'equal', arguments)


        elif function_name == 'LIST' and len(arguments) == 0:
            return self.ivy_func(result, 'new_empty_list', arguments)

        elif function_name == 'LIST' and len(arguments) == 1:
            return self.ivy_func(result, 'new_list_1', arguments)

        elif function_name == 'ITER' and  len(arguments) == 1:
            return self.ivy_func(result, 'new_iter_of', arguments)

        elif function_name == 'BUILTINS.getitem' and len(arguments) == 2:
            return self.ivy_func(result, 'getitem', arguments)

        # The following is not needed after Elazar's fix:
        #
        # elif (function_name == 'BUILTINS.getattr' and
        #       len(arguments) == 2 and
        #       arguments[1] in ["'append'", "'__getitem__'"]
        # ):
        #     func_name = ast.literal_eval(arguments[1]).replace('__','')
        #     return "{} := getattr_{}({})".format(
        #         self.ivy_obj_of(result),
        #         func_name,
        #         self.ivy_obj_of(arguments[0]),
        #     )

        elif function_name == 'TUPLE':
            return "\n".join([
                "tmp_result := *",
                "assume type_of(tmp_result) = t_tuple",
            ] + [
                "assume item(tmp_result,{},X) <-> X = {}".format(
                    self.ivy_obj_of(str(i)),
                    self.ivy_obj_of(arguments[i]),
                )
                for i in range(len(arguments))
            ] + [
                "{} := tmp_result".format(self.ivy_obj_of(result)),
            ])

        else:
            assert False, "builtin_function({!r}, {!r}, {!r})".format(
                result, function_name, arguments
            )

    def block_instructions_to_ivy(self, block_instructions):
        instructions = [self.instruction_to_ivy(instruction) for instruction in block_instructions]
        instructions = [x for x in instructions if x != '']
        instructions = [y for x in instructions for y in x.splitlines()]
        return instructions

    def tac_blocks_cfg_to_ivy(self, tac_blocks_cfg):
        self.python_vars = dict()
        self.python_lits = dict()
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

        python_vars = sorted((v, x) for x, v in self.python_vars.items())
        ivy_python_vars_decls = "\n".join(
            "individual {}:python_object  # {}".format(v, x)
            for v, x in python_vars
        )

        python_lits = sorted((lit, x) for x, lit in self.python_lits.items())
        ivy_python_lits_decls = "\n".join(
            "individual {}:python_object  # {}\n".format(lit, x) +
            "axiom type_of({}) = {}".format(lit, self.ivy_type_of_lit(x))
            for lit, x in python_lits
        )

        return "\n\n\n".join(
            [
                "#lang ivy1.3",
                "\n".join("# " + line for line in tac.cfg_to_lines(tac_blocks_cfg)),
                ivy_type_decls,
                ivy_can_run_decls,
                ivy_python_lits_decls,
                ivy_python_vars_decls,
            ] +
            actions
        )


def test(code_to_analyze):
    from pdb import set_trace
    # set_trace()
    #tac_blocks_cfg = tac_analysis.make_tacblock_cfg(code_to_analyze)
    tac_blocks_cfg = tac.make_tacblock_cfg(code_to_analyze)
    print("\n".join("# " + line for line in tac.cfg_to_lines(tac_blocks_cfg)))
    print()
    tac_to_ivy = TacToIvy()
    result = tac_to_ivy.tac_blocks_cfg_to_ivy(tac_blocks_cfg)
    open("ivy_file.ivy", 'w').write(result)
    #print(result)


if __name__ == "__main__":
    test(test_function)
