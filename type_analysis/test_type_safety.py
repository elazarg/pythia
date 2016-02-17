import type_analysis

def all_tests():
    basic_tests()

def basic_tests():
    assert type_analysis.analyze_type_safety("x = 1; y = x + 1") == True
    assert type_analysis.analyze_type_safety("x = 1; y = x + 'r'") == False
    
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
while True:
    x = 'bla'
z = 'hi' + x """) == False # becuase possibly the while is not executed and x is uninitialized
assert type_analysis.analyze_type_safety("""
x = 'buya'
while True:
    x = 'bla'
z = 'hi' + x """) == True
    
if __name__ == "__main__":
    all_tests()