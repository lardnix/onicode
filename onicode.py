import sys
import os
import subprocess

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

class OniTokenType(Enum):
    NUMBER     = auto()
    IDENTIFIER = auto()
    KEYWORD    = auto()
    SYMBOL     = auto()
    DATATYPE   = auto()
    OPERATOR   = auto()

    EOF        = auto()

class OniKeywordType(Enum):
    FUNC  = auto()
    
    TRUE  = auto()
    FALSE = auto()
    
    NULL  = auto()

class OniSymbolType(Enum):
    L_PAREN = auto()
    R_PAREN = auto()

    L_BRACE = auto()
    R_BRACE = auto()
    
    SEMI    = auto()

    D_MINUS = auto()

class OniDataType(Enum):
    NOTHING = auto()

class OniOperator(Enum):
    PLUS     = auto()
    MINUS    = auto()
    MULTIPLY = auto()
    DIVIDE   = auto()
    MODULO   = auto()
    
    BIT_AND  = auto()
    BIT_OR   = auto()
    BIT_XOR  = auto()
    BIT_SHL  = auto()
    BIT_SHR  = auto()
    BIT_NOT  = auto()
    
    LOG_EQ   = auto()
    LOG_NEQ  = auto()
    LOG_LT   = auto()
    LOG_GT   = auto()
    LOG_LTE  = auto()
    LOG_GTE  = auto()
    LOG_AND  = auto()
    LOG_OR   = auto()
    LOG_NOT  = auto()


ONI_KEYWORDS = {
    "func":  OniKeywordType.FUNC,
    
    "true":  OniKeywordType.TRUE,
    "false": OniKeywordType.FALSE,
    
    "null":  OniKeywordType.NULL,
}

ONI_SYMBOLS = {
    "(":  OniSymbolType.L_PAREN,
    ")":  OniSymbolType.R_PAREN,
    
    "{":  OniSymbolType.L_BRACE,
    "}":  OniSymbolType.R_BRACE,

    ";":  OniSymbolType.SEMI,

    "--": OniSymbolType.D_MINUS,
}

ONI_DATATYPES = {
    "nothing": OniDataType.NOTHING,
}

ONI_OPERATORS = {
    "+":  OniOperator.PLUS,
    "-":  OniOperator.MINUS,
    "*":  OniOperator.MULTIPLY,
    "/":  OniOperator.DIVIDE,
    "%":  OniOperator.MODULO,
    
    "&":  OniOperator.BIT_AND,
    "|":  OniOperator.BIT_OR,
    "^":  OniOperator.BIT_XOR,
    "<<": OniOperator.BIT_SHL,
    ">>": OniOperator.BIT_SHR,
    "~":  OniOperator.BIT_NOT,
    
    "==": OniOperator.LOG_EQ,
    "!=": OniOperator.LOG_NEQ,
    "<":  OniOperator.LOG_LT,
    ">":  OniOperator.LOG_GT,
    "<=": OniOperator.LOG_LTE,
    ">=": OniOperator.LOG_GTE,
    "&&": OniOperator.LOG_AND,
    "||": OniOperator.LOG_OR,
    "!":  OniOperator.LOG_NOT,
}

@dataclass
class OniToken:
    type: OniTokenType
    value: str


@dataclass
class AST:
    pass

@dataclass
class AST_OniRoot(AST):
    global_declarations: List[AST]

@dataclass
class AST_OniFunction_Declaration(AST):
    name: str
    parameters: List[AST]
    return_type: OniDataType
    body: AST

@dataclass
class AST_OniBlock(AST):
    statements: List[AST]

@dataclass
class AST_OniExpression_Statement(AST):
    value: AST

@dataclass
class AST_OniEmpty_Statement(AST):
    pass

@dataclass
class AST_OniBinary_Expression(AST):
    lhs: AST
    operator: OniOperator
    rhs: AST

@dataclass
class AST_OniUnary_Expresstion(AST):
    operator: OniOperator
    value: AST

@dataclass
class AST_OniFunctionCall_Expression(AST):
    name: str
    arguments: List[AST]

@dataclass
class AST_OniNumber_Expression(AST):
    value: int

@dataclass
class AST_OniTrue_Expression(AST):
    pass

@dataclass
class AST_OniFalse_Expression(AST):
    pass

@dataclass
class AST_OniNull_Expression(AST):
    pass

@dataclass
class AST_OniArgument(AST):
    value: AST




def throw_error(message: str):
    print(f"[Error]: {message}")
    exit(1)

class Lexer:
    def __init__(self, src: str):
        self.src = src
        self.pos = 0
        self.c = self.src[self.pos]

    def advance(self):

        if self.pos < len(self.src) - 1:
            self.pos += 1
            self.c = self.src[self.pos]
        else:
            self.c = "\0"

    def skip_comments(self):
        while self.c != "\n":
            self.advance()

    def skip_whitespaces(self):
        while self.c.isspace() and self.c != "\0":
            self.advance()

    def number(self):
        lexeme = ""

        while self.c.isdigit():
            lexeme += self.c
            self.advance()

        return lexeme

    def identifier(self):
        lexeme = ""

        while self.c.isalpha() or self.c.isdigit() or self.c == "-" or self.c == "_":
            lexeme += self.c
            self.advance()

        return lexeme

    def next_token(self):
        while self.c != "\0":
            if self.pos < len(self.src) - 1:
                comment_start = self.c + self.src[self.pos + 1]

                if comment_start == "//":
                    self.skip_comments()
                    continue

            if self.c.isspace():
                self.skip_whitespaces()
                continue

            if self.c.isdigit():
                lexeme = self.number()

                return OniToken(type=OniTokenType.NUMBER, value=lexeme)

            if self.c.isalpha():
                lexeme = self.identifier()

                if lexeme in ONI_KEYWORDS:
                    return OniToken(type=OniTokenType.KEYWORD, value=lexeme)
                if lexeme in ONI_DATATYPES:
                    return OniToken(type=OniTokenType.DATATYPE, value=lexeme)
                
                return OniToken(type=OniTokenType.IDENTIFIER, value=lexeme)
            
            lexeme = self.c
            if lexeme in ONI_SYMBOLS or lexeme + self.src[self.pos + 1] in ONI_SYMBOLS:
                self.advance()

                double_symbol = lexeme + self.c

                if double_symbol in ONI_SYMBOLS:
                    self.advance()
                    return OniToken(type=OniTokenType.SYMBOL, value=double_symbol)

                return OniToken(type=OniTokenType.SYMBOL, value=lexeme)

            elif lexeme in ONI_OPERATORS or lexeme + self.src[self.pos + 1] in ONI_OPERATORS:
                self.advance()

                double_operator = lexeme + self.c;

                if double_operator in ONI_OPERATORS:
                    self.advance()
                    return OniToken(type=OniTokenType.OPERATOR, value=double_operator)

                return OniToken(type=OniTokenType.OPERATOR, value=lexeme)
            
            throw_error(f"Invalid lexeme: {lexeme}")

        return OniToken(type=OniTokenType.EOF, value=self.c)

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer;
        self.token = self.lexer.next_token()

    def eat(self, token_type: OniTokenType, *args) -> OniToken:
        token = self.token

        match token_type:
            case OniTokenType.NUMBER | OniTokenType.IDENTIFIER | OniTokenType.DATATYPE | OniTokenType.EOF:
                if token.type != token_type:
                    throw_error(f"Invalid token type, expectd {token_type} but got {token.type}")
            case OniTokenType.KEYWORD:
                if token.type != token_type:
                    throw_error(f"Invalid token type, expectd {token_type} but got {token.type}")

                token_keyword_type = ONI_KEYWORDS[token.value]
                keyword_type = args[0]

                if token_keyword_type != keyword_type:
                    throw_error(f"Invalid keyword type, expectd {keyword_type} but got {token_keyword_type}")
            case OniTokenType.SYMBOL:
                if token.type != token_type:
                    throw_error(f"Invalid token type, expectd {token_type} but got {token.type}")

                token_symbol_type = ONI_SYMBOLS[token.value]
                symbol_type = args[0]

                if token_symbol_type != symbol_type:
                    throw_error(f"Invalid symbol type, expectd {symbol_type} but got {token_symbol_type}")
            case OniTokenType.OPERATOR:
                if token.type != token_type:
                    throw_error(f"Invalid token type, expectd {token_type} but got {token.type}")

                token_operator_type = ONI_OPERATORS[token.value]
                operator_type = args[0]

                if token_operator_type != operator_type:
                    throw_error(f"Invalid operator type, expectd {operator_type} but got {token_operator_type}")



        self.token = self.lexer.next_token()

        return token


    def argument(self) -> AST_OniArgument:
        return AST_OniArgument(value=self.expression())

    def function_call_expr(self) -> AST_OniFunctionCall_Expression:
        name = self.eat(OniTokenType.IDENTIFIER)
        
        arguments: List[AST] = []

        self.eat(OniTokenType.SYMBOL, OniSymbolType.L_PAREN)
        
        arguments.append(self.argument())

        self.eat(OniTokenType.SYMBOL, OniSymbolType.R_PAREN)

        return AST_OniFunctionCall_Expression(name=name.value, arguments=arguments)

    def number_expr(self) -> AST_OniNumber_Expression:
        return AST_OniNumber_Expression(value=int(self.eat(OniTokenType.NUMBER).value))

    def true_expr(self) -> AST_OniTrue_Expression:
        self.eat(OniTokenType.KEYWORD, OniKeywordType.TRUE)
        return AST_OniTrue_Expression()

    def false_expr(self) -> AST_OniFalse_Expression:
        self.eat(OniTokenType.KEYWORD, OniKeywordType.FALSE)
        return AST_OniFalse_Expression()

    def null_expr(self) -> AST_OniNull_Expression:
        self.eat(OniTokenType.KEYWORD, OniKeywordType.NULL)
        return AST_OniNull_Expression()

    def primary_expr(self) -> AST:
        if self.token.type == OniTokenType.KEYWORD:
            keyword = ONI_KEYWORDS[self.token.value]

            if keyword == OniKeywordType.TRUE: return self.true_expr()
            if keyword == OniKeywordType.FALSE: return self.false_expr()
            if keyword == OniKeywordType.NULL: return self.null_expr()
        
        if self.token.type == OniTokenType.NUMBER:
            return self.number_expr()

        if self.token.type == OniTokenType.SYMBOL and ONI_SYMBOLS[self.token.value] == OniSymbolType.L_PAREN:
            self.eat(OniTokenType.SYMBOL, OniSymbolType.L_PAREN)
            expr = self.expression()
            self.eat(OniTokenType.SYMBOL, OniSymbolType.R_PAREN)

            return expr
        
        return self.function_call_expr()

    def unary_expr(self) -> AST:
        if self.token.type == OniTokenType.OPERATOR:
            operator = ONI_OPERATORS[self.token.value]

            match operator:
                case OniOperator.PLUS | OniOperator.MINUS | OniOperator.BIT_NOT | OniOperator.LOG_NOT:
                    self.eat(OniTokenType.OPERATOR, operator)

                    return AST_OniUnary_Expresstion(operator=operator, value=self.unary_expr())

        return self.primary_expr()

    def bitwise_expr(self) -> AST:
        ast = self.unary_expr()

        while self.token.type == OniTokenType.OPERATOR:
            operator = ONI_OPERATORS[self.token.value]

            match operator:
                case OniOperator.BIT_AND | OniOperator.BIT_OR | OniOperator.BIT_XOR | OniOperator.BIT_SHL | OniOperator.BIT_SHR:
                    self.eat(OniTokenType.OPERATOR, operator)
                    ast = AST_OniBinary_Expression(lhs=ast, operator=operator, rhs=self.unary_expr())
                case _:
                    break;
        return ast

    def factor_expr(self) -> AST:
        ast = self.bitwise_expr()

        while self.token.type == OniTokenType.OPERATOR:
            operator = ONI_OPERATORS[self.token.value]

            match operator:
                case OniOperator.DIVIDE | OniOperator.MULTIPLY | OniOperator.MODULO:
                    self.eat(OniTokenType.OPERATOR, operator)
                    ast = AST_OniBinary_Expression(lhs=ast, operator=operator, rhs=self.bitwise_expr())
                case _:
                    break;
        return ast

    def term_expr(self) -> AST:
        ast = self.factor_expr()

        while self.token.type == OniTokenType.OPERATOR:
            operator = ONI_OPERATORS[self.token.value]

            match operator:
                case OniOperator.PLUS | OniOperator.MINUS:
                    self.eat(OniTokenType.OPERATOR, operator)
                    ast = AST_OniBinary_Expression(lhs=ast, operator=operator, rhs=self.factor_expr())
                case _:
                    break;
        return ast

    def relational_expr(self) -> AST:
        ast = self.term_expr()

        while self.token.type == OniTokenType.OPERATOR:
            operator = ONI_OPERATORS[self.token.value]

            match operator:
                case OniOperator.LOG_LT | OniOperator.LOG_GT | OniOperator.LOG_LTE | OniOperator.LOG_GTE:
                    self.eat(OniTokenType.OPERATOR, operator)
                    ast = AST_OniBinary_Expression(lhs=ast, operator=operator, rhs=self.term_expr())
                case _:
                    break;
        return ast

    def equality_expr(self) -> AST:
        ast = self.relational_expr()

        while self.token.type == OniTokenType.OPERATOR:
            operator = ONI_OPERATORS[self.token.value]

            match operator:
                case OniOperator.LOG_EQ | OniOperator.LOG_NEQ:
                    self.eat(OniTokenType.OPERATOR, operator)
                    ast = AST_OniBinary_Expression(lhs=ast, operator=operator, rhs=self.relational_expr())
                case _:
                    break;
        return ast



    def logical_expr(self) -> AST:
        ast = self.equality_expr()

        while self.token.type == OniTokenType.OPERATOR:
            operator = ONI_OPERATORS[self.token.value]

            match operator:
                case OniOperator.LOG_AND | OniOperator.LOG_OR:
                    self.eat(OniTokenType.OPERATOR, operator)
                    ast = AST_OniBinary_Expression(lhs=ast, operator=operator, rhs=self.equality_expr())
                case _:
                    break;
        return ast

    def assignment_expr(self) -> AST:
        #TODO: Update when include variables
        return self.logical_expr()

    def expression(self) -> AST_OniExpression_Statement:
        return AST_OniExpression_Statement(value=self.assignment_expr())


    def statement(self) -> AST:
        if self.token.type == OniTokenType.IDENTIFIER:
            return self.expression()

        return AST_OniEmpty_Statement()

    def block(self) -> AST_OniBlock:
        self.eat(OniTokenType.SYMBOL, OniSymbolType.L_BRACE)

        statements = [self.statement()]

        while self.token.type == OniTokenType.SYMBOL and ONI_SYMBOLS[self.token.value] == OniSymbolType.SEMI:
            self.eat(OniTokenType.SYMBOL, OniSymbolType.SEMI)
            statements.append(self.statement())

        self.eat(OniTokenType.SYMBOL, OniSymbolType.R_BRACE)

        return AST_OniBlock(statements=statements)

    def function_declaration(self) -> AST_OniFunction_Declaration:
        self.eat(OniTokenType.KEYWORD, OniKeywordType.FUNC)

        name = self.eat(OniTokenType.IDENTIFIER)

        self.eat(OniTokenType.SYMBOL, OniSymbolType.L_PAREN)
        self.eat(OniTokenType.SYMBOL, OniSymbolType.R_PAREN)

        self.eat(OniTokenType.SYMBOL, OniSymbolType.D_MINUS)

        return_type = self.eat(OniTokenType.DATATYPE)

        body = self.block()

        return AST_OniFunction_Declaration(name=name.value, parameters=[], return_type=ONI_DATATYPES[return_type.value], body=body)

    def global_declaration(self) -> AST:
        if self.token.type == OniTokenType.KEYWORD and ONI_KEYWORDS[self.token.value] == OniKeywordType.FUNC:
            return self.function_declaration()
        
        throw_error(f"Expected keyword func but got {self.token.value}")
        return AST()

    def root(self) -> AST_OniRoot:
        ast_root = AST_OniRoot(global_declarations=[])

        while self.token.type != OniTokenType.EOF:
            ast_root.global_declarations.append(self.global_declaration())

        return ast_root

    def parse(self) -> AST:
        ast = self.root()
        return ast

class Generator:
    def __init__(self, parser: Parser):
        self.parser = parser

        self.src = ""

        self.argument_count = 0

    def get_argument_register(self):
        registers = ["rdi", "rsi", "rdx", "r10", "r8", "r9"]

        if self.argument_count < len(registers):
            return registers[self.argument_count]


    def gen_AST_OniArgument(self, ast: AST_OniArgument):
        self.generate(ast.value)

        self.src += f"  mov {self.get_argument_register()}, rax\n"

    def gen_AST_OniFunctionCall_Expression(self, ast: AST_OniFunctionCall_Expression):
        self.argument_count = 0

        for argument in ast.arguments:
            self.generate(argument)

        self.src += f"  call {ast.name}\n"

    def gen_AST_OniEmpty_Statement(self, ast: AST_OniEmpty_Statement):
        pass

    def gen_AST_OniNumber_Expression(self, ast: AST_OniNumber_Expression):
        self.src += f"  mov rax, {ast.value}\n"

    def gen_AST_OniTrue_Expression(self, ast: AST_OniTrue_Expression):
        self.src += "  mov rax, 1\n"

    def gen_AST_OniFalse_Expression(self, ast: AST_OniFalse_Expression):
        self.src += "  mov rax, 0\n"

    def gen_AST_OniNull_Expression(self, ast: AST_OniNull_Expression):
        self.src += "  mov rax, 0\n"

    def gen_AST_OniUnary_Expresstion(self, ast: AST_OniUnary_Expresstion):
        self.generate(ast.value)

        match ast.operator:
            case OniOperator.PLUS:
                self.src += "  xor rbx, rbx\n"
                self.src += "  xchg rax, rbx\n"
                self.src += "  add rax, rbx\n"
            case OniOperator.MINUS:
                self.src += "  xor rbx, rbx\n"
                self.src += "  xchg rax, rbx\n"
                self.src += "  sub rax, rbx\n"
            case OniOperator.BIT_NOT:
                self.src += "  not rax\n"
            case OniOperator.LOG_NOT:
                self.src += "  mov rcx, 0\n"
                self.src += "  mov rdx, 1\n"
                self.src += "  cmp rax, 0\n"
                self.src += "  cmove rcx, rdx\n"
                self.src += "  mov rax, rcx\n"

    def gen_AST_OniBinary_Expression(self, ast: AST_OniBinary_Expression):
        self.generate(ast.lhs)
        self.src += "  push rax\n"
        self.generate(ast.rhs)
        self.src += "  pop rbx\n"
        self.src += "  xchg rax, rbx\n"

        match ast.operator:
            case OniOperator.PLUS:
                self.src += "  add rax, rbx\n"
            case OniOperator.MINUS:
                self.src += "  sub rax, rbx\n"
            case OniOperator.MULTIPLY:
                self.src += "  xor rdx, rdx\n"
                self.src += "  mul rbx\n"
            case OniOperator.DIVIDE:
                self.src += "  xor rdx, rdx\n"
                self.src += "  div rbx\n"
            case OniOperator.MODULO:
                self.src += "  xor rdx, rdx\n"
                self.src += "  div rbx\n"
                self.src += "  mov rax, rdx\n"
            
            case OniOperator.BIT_AND:
                self.src += "  and rax, rbx\n"
            case OniOperator.BIT_OR:
                self.src += "  or rax, rbx\n"
            case OniOperator.BIT_XOR:
                self.src += "  xor rax, rbx\n"
            case OniOperator.BIT_SHL:
                self.src += "  mov rcx, rbx\n"
                self.src += "  shl rax, cl\n"
            case OniOperator.BIT_SHR:
                self.src += "  mov rcx, rbx\n"
                self.src += "  shr rax, cl\n"
            
            case OniOperator.LOG_EQ:
                self.src += "  mov rcx, 0\n"
                self.src += "  mov rdx, 1\n"
                self.src += "  cmp rax, rbx\n"
                self.src += "  cmove rcx, rdx\n"
                self.src += "  mov rax, rcx\n"
            case OniOperator.LOG_NEQ:
                self.src += "  mov rcx, 0\n"
                self.src += "  mov rdx, 1\n"
                self.src += "  cmp rax, rbx\n"
                self.src += "  cmovne rcx, rdx\n"
                self.src += "  mov rax, rcx\n"
            case OniOperator.LOG_LT:
                self.src += "  mov rcx, 0\n"
                self.src += "  mov rdx, 1\n"
                self.src += "  cmp rax, rbx\n"
                self.src += "  cmovl rcx, rdx\n"
                self.src += "  mov rax, rcx\n"
            case OniOperator.LOG_GT:
                self.src += "  mov rcx, 0\n"
                self.src += "  mov rdx, 1\n"
                self.src += "  cmp rax, rbx\n"
                self.src += "  cmovg rcx, rdx\n"
                self.src += "  mov rax, rcx\n"
            case OniOperator.LOG_LTE:
                self.src += "  mov rcx, 0\n"
                self.src += "  mov rdx, 1\n"
                self.src += "  cmp rax, rbx\n"
                self.src += "  cmovle rcx, rdx\n"
                self.src += "  mov rax, rcx\n"
            case OniOperator.LOG_GTE:
                self.src += "  mov rcx, 0\n"
                self.src += "  mov rdx, 1\n"
                self.src += "  cmp rax, rbx\n"
                self.src += "  cmovge rcx, rdx\n"
                self.src += "  mov rax, rcx\n"
            case OniOperator.LOG_AND:
                self.src += "  and rax, rbx\n"
            case OniOperator.LOG_OR:
                self.src += "  or rax, rbx\n"



    def gen_AST_OniExpression_Statement(self, ast: AST_OniExpression_Statement):
        self.generate(ast.value)

    def gen_AST_OniBlock(self, ast: AST_OniBlock):
        for statement in ast.statements:
            self.generate(statement)

    def gen_AST_OniFunction_Declaration(self, ast: AST_OniFunction_Declaration):
        self.src += f"{ast.name}:\n"

        self.generate(ast.body)

        self.src += "  xor rax, rax\n"
        self.src += "  ret\n"

    def gen_AST_OniRoot(self, ast: AST_OniRoot):
        for global_decl in ast.global_declarations:
            self.generate(global_decl)

    def default_gen(self, ast: AST):
        ast_name = type(ast).__name__

        throw_error(f"Generator for {ast_name} not exists")

    def generate(self, ast: AST):
        method_name = f"gen_{type(ast).__name__}"

        method = getattr(self, method_name, self.default_gen)

        method(ast)

    def gen(self) -> str:
        ast = self.parser.parse()

        self.src += "format ELF64 executable 3\n"
        self.src += "entry start\n"
        self.src += "segment readable executable\n"
        self.src += "start:\n"
        self.src += "  call main\n"
        self.src += "  mov rax, 60\n"
        self.src += "  mov rdi, 0\n"
        self.src += "  syscall\n"
        self.src += "putd:\n"
        self.src += "  mov rax, rdi\n"
        self.src += "  mov rbx, 10\n"
        self.src += "  mov rcx, putd_buffer\n"
        self.src += "  mov [rcx], rbx\n"
        self.src += "  inc rcx\n"
        self.src += "  mov QWORD [putd_buffer_pos], rcx\n"
        self.src += ".push_digits:\n"
        self.src += "  xor rdx, rdx\n"
        self.src += "  mov rbx, 10\n"
        self.src += "  div rbx\n"
        self.src += "  push rax\n"
        self.src += "  add rdx, 48\n"
        self.src += "  mov rcx, QWORD [putd_buffer_pos]\n"
        self.src += "  mov [rcx], dl\n"
        self.src += "  inc rcx\n"
        self.src += "  mov QWORD [putd_buffer_pos], rcx\n"
        self.src += "  pop rax\n"
        self.src += "  cmp rax, 0\n"
        self.src += "  jne .push_digits\n"
        self.src += ".print_digits:\n"
        self.src += "  mov rcx, QWORD [putd_buffer_pos]\n"
        self.src += "  mov rax, 1\n"
        self.src += "  mov rdi, 1\n"
        self.src += "  mov rsi, rcx\n"
        self.src += "  mov rdx, 1\n"
        self.src += "  syscall\n"
        self.src += "  mov rcx, QWORD [putd_buffer_pos]\n"
        self.src += "  dec rcx\n"
        self.src += "  mov QWORD [putd_buffer_pos], rcx\n"
        self.src += "  cmp rcx, putd_buffer\n"
        self.src += "  jge .print_digits\n"
        self.src += "  ret\n"

        self.generate(ast)

        self.src += "segment readable writable\n"
        self.src += "putd_buffer rb 100\n"
        self.src += "putd_buffer_pos rb 8\n"


        return self.src
def usage(fd, program):
    print(f"[Usage]: {program} <FLAGS> <input/path.oni>", file=fd)
    print(f"  [FLAGS]", file=fd)
    print(f"    -o/--output <output/path>                      - set the output path", file=fd)
    print(f"    -h/--help                                      - show this usage message", file=fd)

def main():
    program_name, *argv = sys.argv

    if len(argv) <= 0:
        usage(sys.stderr, program_name)
        throw_error("Input file path is not provided")

    input_path = None
    output_path = None
    while len(argv) > 0:
        arg, *argv = argv

        match arg:
            case "-o" | "--output":
                if len(argv) <= 0:
                    usage(sys.stderr, program_name)
                    throw_error("-o/--output argument is not provided")

                output_path, *argv = argv

            case "-h" | "--help":
                usage(sys.stdout, program_name)
                exit()

            case _:
                if input_path != None:
                    usage(sys.stderr, program_name)
                    throw_error(f"Unknown flag: {arg}")

                input_path = arg
    
    if input_path == None:
        usage(sys.stderr, program_name)
        throw_error("Input file path is not provided")

    assert input_path != None

    if output_path == None:
        output_path = os.path.splitext(input_path)[0]

    if not os.path.exists(input_path):
        throw_error(f"File {input_path} not exists")

    src = open(input_path).read()

    if len(src) <= 0:
        print("[Note]: Input file is empty, exiting...")
        exit()

    lexer = Lexer(src)
    parser = Parser(lexer)
    generator = Generator(parser)

    assembly = generator.gen()

    with open(output_path + ".asm", "w") as output:
        output.write(assembly)

    subprocess.call(["fasm", output_path + ".asm"])


if __name__ == "__main__":
    main()
