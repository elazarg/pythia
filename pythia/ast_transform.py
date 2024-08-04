from __future__ import annotations as _

import ast
import typing

import black


def annotated_for_labels(node: ast.FunctionDef) -> frozenset[int]:
    return frozenset(
        n.lineno for n in ast.walk(node) if isinstance(n, ast.For) and n.type_comment
    )


class Parser:
    filename: str

    def __init__(self, filename: str) -> None:
        self.filename = filename

    def parse(self, source: str) -> ast.Module:
        return ast.parse(
            source, type_comments=True, filename=self.filename, feature_version=(3, 12)
        )

    def parse_statement(self, source: str) -> ast.stmt:
        stmt = self.parse(source).body[0]
        assert isinstance(stmt, ast.stmt)
        return stmt

    def parse_expression(self, source: str) -> ast.expr:
        stmt = self.parse(source).body[0]
        assert isinstance(stmt, ast.Expr)
        return stmt.value

    def iterate_purified_functions(
        self, node: ast.Module
    ) -> typing.Iterator[ast.FunctionDef]:
        for node in node.body:
            if isinstance(node, ast.FunctionDef):
                for defaults in [node.args.defaults, node.args.kw_defaults]:
                    for i, v in enumerate(defaults):
                        v = ast.literal_eval(v)
                        defaults[i] = self.parse_expression(repr(v))
                yield node


def make_assign(variables: typing.Iterable[str], value: ast.expr):
    return ast.Assign(
        targets=[
            ast.List(
                elts=[ast.Name(id=x, ctx=ast.Store()) for x in variables],
                ctx=ast.Store(),
            )
        ],
        value=value,
    )


def make_for(for_loop: ast.For, filename: str, _dirty: set[str]) -> ast.With:
    dirty = tuple(sorted(_dirty))
    parse_expression = Parser(filename).parse_expression
    iter = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="transaction", ctx=ast.Load()),
            attr="iterate",
            ctx=ast.Load(),
        ),
        args=[for_loop.iter],
        keywords=[],
    )

    for_loop = ast.For(
        target=for_loop.target,
        iter=iter,
        body=[
            *for_loop.body,
            ast.Expr(parse_expression(f'transaction.commit({", ".join(dirty)})')),
        ],
        orelse=for_loop.orelse,
        type_comment=for_loop.type_comment,
        lineno=for_loop.lineno,
        col_offset=for_loop.col_offset,
    )

    res = ast.With(
        items=[
            ast.withitem(
                context_expr=parse_expression("persist.Loader(__file__, locals())"),
                optional_vars=ast.Name(id="transaction", ctx=ast.Store()),
            )
        ],
        body=[
            ast.If(
                test=parse_expression("transaction"),
                body=[make_assign(dirty, parse_expression("transaction.move()"))],
                orelse=[],
            ),
            for_loop,
        ],
    )
    return res


class VariableFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.dirty_map: dict[str, dict[int, set[str]]] = {}

    def visit_FunctionDef(self, func: ast.FunctionDef) -> None:
        dirty = {node.arg for node in ast.walk(func.args) if isinstance(node, ast.arg)}
        dirty |= {
            node.id
            for node in ast.walk(func)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.dirty_map[func.name] = {
            node.lineno: dirty
            for node in ast.walk(func)
            if isinstance(node, ast.For) and node.type_comment
        }


class DirtyTransformer(ast.NodeTransformer):
    def __init__(self, filename: str, dirty_per_line: dict[int, set[str]]) -> None:
        self.filename = filename
        self.dirty_per_line = dirty_per_line

    def visit_For(self, for_loop: ast.For) -> ast.With | ast.For:
        for_loop = typing.cast(ast.For, self.generic_visit(for_loop))
        assert isinstance(for_loop, ast.For)
        if not for_loop.type_comment:
            return for_loop
        return make_for(for_loop, self.filename, self.dirty_per_line[for_loop.lineno])


class Compiler(ast.NodeTransformer):
    def __init__(
        self,
        filename: str,
        dirty_map: dict[str, dict[int, set[str]]],
        parser: Parser,
        naive: bool = False,
    ) -> None:
        self.filename = filename
        self.dirty_map = dirty_map
        self.parser = parser
        self.naive = naive

    def visit_Module(self, node: ast.Module) -> ast.Module:
        tree = typing.cast(ast.Module, self.generic_visit(node))
        import_stmt = self.parser.parse_statement("from checkpoint import persist")
        res = ast.Module(body=[import_stmt, *tree.body], type_ignores=tree.type_ignores)
        return res

    def visit_FunctionDef(self, function: ast.FunctionDef) -> ast.FunctionDef:
        dirty = self.dirty_map.get(function.name)
        if not dirty:
            return function
        res = DirtyTransformer(self.filename, dirty).visit(function)
        assert isinstance(res, ast.FunctionDef)
        if self.naive:
            all_vars = set.union(*dirty.values())
            args = {
                node.arg for node in ast.walk(res.args) if isinstance(node, ast.arg)
            }
            locals = tuple(sorted(all_vars - args))
            initialize = make_assign(
                locals, self.parser.parse_expression(f"(None,)*{len(locals)}")
            )
            res.body.insert(0, initialize)
        return res


def transform(
    filename: str, dirty_map: typing.Optional[dict[str, dict[int, set[str]]]]
) -> str:
    parser = Parser(filename)

    with open(filename, encoding="utf-8") as f:
        source = f.read()
    tree = parser.parse(source)

    naive = dirty_map is None
    if naive:
        finder = VariableFinder()
        finder.visit(tree)
        dirty_map = finder.dirty_map

    tree = Compiler(filename, dirty_map, parser, naive=naive).visit(tree)
    tree = ast.fix_missing_locations(tree)
    res = ast.unparse(tree)
    return black.format_str(res, mode=black.FileMode())


def make_for_tcp(for_loop: ast.For, tag: str, filename: str) -> ast.With:
    parse_expression = Parser(filename).parse_expression
    iter = ast.Call(
        func=parse_expression(f"client.iterate"),
        args=[for_loop.iter],
        keywords=[],
    )

    for_loop = ast.For(
        target=for_loop.target,
        iter=iter,
        body=[
            *for_loop.body,
            ast.Expr(parse_expression(f"client.commit()")),
        ],
        orelse=for_loop.orelse,
        type_comment=for_loop.type_comment,
        lineno=for_loop.lineno,
        col_offset=for_loop.col_offset,
    )

    res = ast.With(
        items=[
            ast.withitem(
                context_expr=parse_expression(f'persist.SimpleTcpClient("{tag}")'),
                optional_vars=ast.Name(id="client", ctx=ast.Store()),
            )
        ],
        body=[
            for_loop,
        ],
    )
    return res


class TcpTransformer(ast.NodeTransformer):
    def __init__(self, tag: str, filename: str) -> None:
        self.tag = tag
        self.filename = filename

    def visit_For(self, for_loop: ast.For) -> ast.With | ast.For:
        for_loop = typing.cast(ast.For, self.generic_visit(for_loop))
        assert isinstance(for_loop, ast.For)
        if not for_loop.type_comment:
            return for_loop
        return make_for_tcp(for_loop, self.tag, self.filename)


class TcpCompiler(ast.NodeTransformer):
    def __init__(
        self, tag: str, filename: str, function_name: str, parser: Parser
    ) -> None:
        self.tag = tag
        self.function_name = function_name
        self.filename = filename
        self.parser = parser

    def visit_Module(self, node: ast.Module) -> ast.Module:
        tree = typing.cast(ast.Module, self.generic_visit(node))
        import_stmt = self.parser.parse_statement("from checkpoint import persist")
        res = ast.Module(body=[import_stmt, *tree.body], type_ignores=tree.type_ignores)
        return res

    def visit_FunctionDef(self, function: ast.FunctionDef) -> ast.FunctionDef:
        if function.name != self.function_name:
            return function
        res = TcpTransformer(self.tag, self.filename).visit(function)
        assert isinstance(res, ast.FunctionDef)
        return res


def tcp_client(tag: str, filename: str, function_name: str) -> str:
    parser = Parser(filename)

    with open(filename, encoding="utf-8") as f:
        source = f.read()
    tree = parser.parse(source)

    tree = TcpCompiler(tag, filename, function_name, parser).visit(tree)
    tree = ast.fix_missing_locations(tree)
    res = ast.unparse(tree)
    return black.format_str(res, mode=black.FileMode())
