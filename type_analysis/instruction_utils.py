import ast
import tac

def __parse_literal_expression(string_expr):
    if tac.is_stackvar(string_expr):
        return ast.Name
    body = ast.parse(string_expr).body
    assert len(body) == 1
    expr = body[0]
    return type(expr.value)

def is_var_reference(string_expr):
    return __parse_literal_expression(string_expr) == ast.Name

def is_numeric_literal(string_expr):
    return __parse_literal_expression(string_expr) == ast.Num

def is_string_literal(string_expr):
    return __parse_literal_expression(string_expr) == ast.Str

def is_bool_literal(string_expr):
    return string_expr.strip() in ['True', 'False']