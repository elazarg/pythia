from __future__ import annotations as _

import ast
import typing


class DirtyTransformer(ast.NodeTransformer):
    def __init__(self, filename: str, function_name: str, dirty: set[str]):
        self.filename = filename
        self.function_name = function_name
        self.dirty = dirty

    def parse_expression(self, source: str) -> ast.expr:
        print(source)
        stmt = ast.parse(source, type_comments=True, filename=self.filename, feature_version=(3, 11)).body[0]
        assert isinstance(stmt, ast.Expr)
        return stmt.value

    def visit_FunctionDef(self, function: ast.FunctionDef) -> ast.FunctionDef:
        if function.name != self.function_name:
            return function
        res = self.generic_visit(function)
        assert isinstance(res, ast.FunctionDef)
        return res

    def visit_For(self, for_loop: ast.For) -> ast.With | ast.For:
        for_loop = typing.cast(ast.For, self.generic_visit(for_loop))
        assert isinstance(for_loop, ast.For)
        if not for_loop.type_comment:
            return for_loop
        iter = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='transaction', ctx=ast.Load()),
                attr='iterate',
                ctx=ast.Load()),
            args=[for_loop.iter],
            keywords=[],
        )
        commit = ast.Expr(
            self.parse_expression(f'transaction.commit({", ".join(self.dirty)})')
        )
        body = [
            *for_loop.body,
            commit
        ]

        for_loop = ast.For(for_loop.target, iter, body, for_loop.orelse, for_loop.type_comment,
                           lineno=for_loop.lineno, col_offset=for_loop.col_offset)

        res = ast.With(
            items=[
                ast.withitem(
                    context_expr=self.parse_expression('persist.Loader(__file__)'),
                    optional_vars=ast.Name(id='transaction', ctx=ast.Store()))],
            body=[
                ast.If(
                    test=self.parse_expression('transaction.restored_state'),
                    body=[
                        ast.Assign(
                            targets=[
                                ast.List(
                                    elts=[ast.Name(id=x, ctx=ast.Store())
                                          for x in self.dirty],
                                    ctx=ast.Store())],
                            value=self.parse_expression('transaction.restored_state'))
                    ],
                    orelse=[]),
                for_loop,
            ]
        )
        return res


def transform(filename: str, function_name: str, dirty: set[str]):
    with open(filename, encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source, type_comments=True, filename=filename, feature_version=(3, 11))
    tree = DirtyTransformer(filename, function_name, dirty).visit(tree)
    tree = ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def read_transformed(filename: str):
    with open(filename, encoding='utf-8') as f:
        source = f.read()
    return ast.parse(source, type_comments=True, filename=filename, feature_version=(3, 11))

if __name__ == '__main__':
    filename = 'examples/toy-instrumented.py'
    print(ast.dump(read_transformed(filename), indent=4))
    print(transform('examples/toy.py', "minimal", {'res'}))
