import os
from instruction import get_instructions
from collections import Counter

def main():
    with open('example.py') as f:
        d = f.read()
    inss = get_instructions(d)
    for i in inss:
        print(i)
    c = Counter(x.opname for x in inss)
    print(c)

def print_instructions(f):
    inss = get_instructions(f)
    for i in inss:
        print(i)
        
def print_stats(path):
    instructions = []
    total = [Counter(),Counter(),Counter(),Counter()] 
    
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.py') and '__' not in filename:
                print(filename)
                with open(dirpath + '/' + filename, encoding='utf8') as f:
                    ins = get_instructions(compile(f.read(), filename, 'exec'))
                    names = [x.opname for x in ins if 'IMPORT' not in x.opname]
                    total[0].update(names)
                    total[1].update(zip(names, names[1:]))
                    total[2].update(zip(names, names[1:], names[2:]))
                    total[3].update(zip(names, names[1:], names[2:], names[3:]))
                    instructions.extend(ins)
    with open('instructions.txt', 'w', encoding='utf8') as f:
        for i in instructions:
            print(i, file=f)
    for n in [0,1,2,3]:
        with open('statistics_{}.txt'.format(n+1), 'w', encoding='utf8') as f:
            for k, v in sorted(total[n].items(), key=lambda x: x[1], reverse=True):
                print(k, round(v/len(instructions), 3), v, sep='; ', file=f)
if __name__ == '__main__':
    print_stats('real_code/django-1.10')
    #compileall.compile_dir('real_code/django-1.10')
    #compileall.compile_dir('real_code/Twisted-15.5.0')

    # print(get_instructions(read_pyc('real_code/django-1.10/__pycache__/setup.cpython-35.pyc')))
'''
LOAD_CONST, 0.343, 51262
STORE_NAME, 0.154, 22965
LOAD_NAME, 0.089, 13294
IMPORT_FROM, 0.081, 12122
CALL_FUNCTION, 0.069, 10254
IMPORT_NAME, 0.059, 8811
POP_TOP, 0.055, 8199
MAKE_FUNCTION, 0.046, 6864
LOAD_BUILD_CLASS, 0.036, 5353
LOAD_ATTR, 0.028, 4176
RETURN_VALUE, 0.015, 2230
BUILD_LIST, 0.006, 931
BUILD_TUPLE, 0.005, 813
BUILD_MAP, 0.004, 572
JUMP_FORWARD, 0.002, 235
POP_JUMP_IF_FALSE, 0.002, 225
COMPARE_OP, 0.001, 181
POP_BLOCK, 0.001, 120
STORE_ATTR, 0.001, 106
DUP_TOP, 0.001, 99
'''
#main()