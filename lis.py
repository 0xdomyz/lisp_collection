################ Lispy: Scheme Interpreter in Python

## See http://norvig.com/lispy.html

import math
import operator as op

Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (int, float)     # A Scheme Number is implemented as a Python int or float
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = list             # A Scheme List is implemented as a Python list
Exp    = (Atom, List)     # A Scheme expression is an Atom or List
Env    = dict             # A Scheme environment (defined below) 
                          # is a mapping of {variable: value}

def parse(program: str) -> Exp:
    """Read a Scheme expression from a string.
    
    Examples
    ----------
    >>> parse("(begin (define r 10) (* pi (* r r)))")
    ['begin', ['define', 'r', 10], ['*', 'pi', ['*', 'r', 'r']]]
    """
    return read_from_tokens(tokenize(program))

def tokenize(chars: str) -> list:
    """Convert a string into a list of tokens.
    
    Examples
    --------------
    >>> tokenize("(begin (define r 10) (* pi (* r r)))")
    ['(', 'begin', '(', 'define', 'r', '10', ')', '(', '*', 'pi', '(', '*', 'r', 'r', ')', ')', ')']
    """
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()

def read_from_tokens(tokens: list) -> Exp:
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF')
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0) # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token: str) -> Atom:
    """Numbers become numbers; every other token is a symbol."""
    try: return int(token)
    except ValueError:
        try: return float(token)
        except ValueError:
            return Symbol(token)

def standard_env() -> Env:
    """An environment with some Scheme standard procedures."""
    env = Env()
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
        'append':  op.add,  
        'apply':   lambda proc, args: proc(*args),
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'expt':    pow,
        'equal?':  op.eq, 
        'length':  len, 
        'list':    lambda *x: List(x), 
        'list?':   lambda x: isinstance(x, List), 
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, Number),  
		'print':   print,
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
    })
    return env

class Env(dict):
    """An environment: a dict of {'var': val} pairs, with an outer Env."""
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

class Procedure(object):
    """A user-defined Scheme procedure."""
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        return eval(self.body, Env(self.parms, args, self.env))

global_env = standard_env()

def eval(x, env=global_env):
    """Evaluate an expression in an environment.
    
    Examples
    --------------
    >>> eval(parse("(begin (define r 10) (* pi (* r r)))"))
    314.1592653589793
    """
    if isinstance(x, Symbol):    # variable reference
        return env.find(x)[x]
    elif not isinstance(x, List):# constant 
        return x   
    op, *args = x       
    if op == 'quote':            # quotation
        return args[0]
    elif op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif op == 'define':         # definition
        (symbol, exp) = args
        env[symbol] = eval(exp, env)
    elif op == 'set!':           # assignment
        (symbol, exp) = args
        env.find(symbol)[symbol] = eval(exp, env)
    elif op == 'lambda':         # procedure
        (parms, body) = args
        return Procedure(parms, body, env)
    else:                        # procedure call
        proc = eval(op, env)
        vals = [eval(arg, env) for arg in args]
        return proc(*vals)

def repl(prompt='lis.py> '):
    """A prompt-read-eval-print loop.
    
    Examples
    ------------
    >>> repl()
    lis.py> (define r 10)
    lis.py> (* pi (* r r))
    314.159265359
    lis.py> (if (> (* 11 11) 120) (* 7 6) oops)
    42
    lis.py> (list (+ 1 1) (+ 2 2) (* 2 3) (expt 2 3))

    >>> repl()
    lis.py> (define circle-area (lambda (r) (* pi (* r r))))
    lis.py> (circle-area 3)
    28.274333882308138
    lis.py> (define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))
    lis.py> (fact 10)
    3628800
    lis.py> (fact 100)
    93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
    lis.py> (circle-area (fact 10))
    41369087205782.695
    lis.py> (define first car)
    lis.py> (define rest cdr)
    lis.py> (define count (lambda (item L) (if L (+ (equal? item (first L)) (count item (rest L))) 0))) 
    lis.py> (count 0 (list 0 1 2 3 0 0))
    3
    lis.py> (count (quote the) (quote (the more the merrier the bigger the better)))
    4
    lis.py> (define twice (lambda (x) (* 2 x)))
    lis.py> (twice 5)
    10
    lis.py> (define repeat (lambda (f) (lambda (x) (f (f x)))))
    lis.py> ((repeat twice) 10)
    40
    lis.py> ((repeat (repeat twice)) 10)
    160
    lis.py> ((repeat (repeat (repeat twice))) 10)
    2560
    lis.py> ((repeat (repeat (repeat (repeat twice)))) 10)
    655360
    lis.py> (pow 2 16)
    65536.0
    lis.py> (define fib (lambda (n) (if (< n 2) 1 (+ (fib (- n 1)) (fib (- n 2))))))
    lis.py> (define range (lambda (a b) (if (= a b) (quote ()) (cons a (range (+ a 1) b)))))
    lis.py> (range 0 10)
    (0 1 2 3 4 5 6 7 8 9)
    lis.py> (map fib (range 0 10))
    <map object at 0x000002AAA4C23E50>
    lis.py> (map fib (range 0 20))
    <map object at 0x000002AAA4F19D90>
    """
    while True:
        val = eval(parse(input(prompt)))
        if val is not None: 
            print(lispstr(val))

def lispstr(exp):
    """Convert a Python object back into a Lisp-readable string."""
    if isinstance(exp, List):
        return '(' + ' '.join(map(lispstr, exp)) + ')' 
    else:
        return str(exp)