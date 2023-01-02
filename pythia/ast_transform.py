from __future__ import annotations as _

import ast
import typing


def make_for(for_loop: ast.For, filename: str, dirty: set[str]) -> ast.With:
    def parse_expression(source: str) -> ast.expr:
        stmt = ast.parse(source, type_comments=True, filename=filename, feature_version=(3, 11)).body[0]
        assert isinstance(stmt, ast.Expr)
        return stmt.value

    iter = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='transaction', ctx=ast.Load()),
            attr='iterate',
            ctx=ast.Load()),
        args=[for_loop.iter],
        keywords=[],
    )
    commit = ast.Expr(
        parse_expression(f'transaction.commit({", ".join(dirty)})')
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
                context_expr=parse_expression('persist.Loader(__file__)'),
                optional_vars=ast.Name(id='transaction', ctx=ast.Store()))],
        body=[
            ast.If(
                test=parse_expression('transaction.restored_state'),
                body=[
                    ast.Assign(
                        targets=[
                            ast.List(
                                elts=[ast.Name(id=x, ctx=ast.Store())
                                      for x in dirty],
                                ctx=ast.Store())],
                        value=parse_expression('transaction.restored_state'))
                ],
                orelse=[]),
            for_loop,
        ]
    )
    return res


def transform(filename: str, dirty_map: dict[str, set[str]]) -> str:
    class DirtyTransformer(ast.NodeTransformer):
        def __init__(self, dirty: set[str]) -> None:
            self.dirty = dirty

        def visit_For(self, for_loop: ast.For) -> ast.With | ast.For:
            for_loop = typing.cast(ast.For, self.generic_visit(for_loop))
            assert isinstance(for_loop, ast.For)
            if not for_loop.type_comment:
                return for_loop
            return make_for(for_loop, filename, self.dirty)

    class Compiler(ast.NodeTransformer):
        def visit_FunctionDef(self, function: ast.FunctionDef) -> ast.FunctionDef:
            if function.name not in dirty_map:
                return function
            res = DirtyTransformer(dirty_map[function.name]).visit(function)
            assert isinstance(res, ast.FunctionDef)
            return res

    with open(filename, encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source, type_comments=True, filename=filename, feature_version=(3, 11))
    tree = Compiler().visit(tree)
    tree = ast.fix_missing_locations(tree)
    return ast.unparse(tree)


if __name__ == '__main__':
    print(transform('examples/toy.py', {"minimal": {'res'}}))
