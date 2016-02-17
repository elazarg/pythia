import type_analysis

def all_tests():
    basic_tests()

def basic_tests():
    # basic addition with variables
    assert type_analysis.analyze_type_safety("x = 1; y = 1; z = x + y") == True
    assert type_analysis.analyze_type_safety("x = 'r'; y = 'r'; z = x + y") == True
    assert type_analysis.analyze_type_safety("x = 1; y = 'r'; z = x + y") == False
    assert type_analysis.analyze_type_safety("x = 'r'; y = 1; z = x + y") == False
    
    # basic addition with literal
    assert type_analysis.analyze_type_safety("x = 1; y = x + 1") == True
    assert type_analysis.analyze_type_safety("x = 1; y = 1 + x") == True
    assert type_analysis.analyze_type_safety("x = 1; y = x + 'r'") == False
    assert type_analysis.analyze_type_safety("x = 1; y = 'r' + x") == False
    assert type_analysis.analyze_type_safety("x = 1; y = 1 + 1") == True
    assert type_analysis.analyze_type_safety("x = 1; y = 'r' + 'r'") == True
    
    # branches
    assert type_analysis.analyze_type_safety("""
if True:
    x = 1
else:
    x = 'bla'
z = 'hi' + x """) == False

    assert type_analysis.analyze_type_safety("""
if True:
    x = 'bla'
else:
    x = 'bla2'
z = 'hi' + x """) == True

    assert type_analysis.analyze_type_safety("""
if True:
    x = 1
else:
    x = 2
z = 2 + x """) == True
    assert type_analysis.analyze_type_safety("""
if True:
    x = 'hi'
else:
    x = 'hi'
z = 2 + x """) == False
    assert type_analysis.analyze_type_safety("""
if True:
    x = 'hi'
else:
    x = 1
z = 2 + x """) == False

    assert type_analysis.analyze_type_safety("""
if True:
    x = 1
    y = 1
else:
    x = 'bla'
    y = 'bla2'
z = x + y""") == True

    # while
    assert type_analysis.analyze_type_safety("""
while True:
    x = 'bla'
z = 'hi' + x """) == False # because possibly the while is not executed and x is uninitialized
    assert type_analysis.analyze_type_safety("""
x = 'buya'
while True:
    x = 'bla'
z = 'hi' + x """) == True

# This is a bad test: the compiler recognizes that the loop is infinite, but not that it must not be entered
#     assert type_analysis.analyze_type_safety("""
# y = 1
# while True:
#     y = 'hello'
# z = 1 + y""") == False

    # type exclusion
    
    
if __name__ == "__main__":
    all_tests()