#pragma once

// Python 3.13 Bytecode Opcodes
// Reference: https://docs.python.org/3.13/library/dis.html

namespace justjit
{
    namespace op
    {

        // Stack manipulation
        constexpr int CACHE = 0;
        constexpr int POP_TOP = 32;
        constexpr int PUSH_NULL = 34;
        constexpr int NOP = 30;
        constexpr int COPY = 61;
        constexpr int SWAP = 115;

        // Unary operations
        constexpr int UNARY_INVERT = 41;
        constexpr int UNARY_NEGATIVE = 42;
        constexpr int UNARY_NOT = 43;
        constexpr int TO_BOOL = 40;

        // Binary operations
        constexpr int BINARY_OP = 45;
        constexpr int BINARY_SUBSCR = 5;
        constexpr int BINARY_SLICE = 4;

        // Compare operations
        constexpr int COMPARE_OP = 58;
        constexpr int CONTAINS_OP = 59;
        constexpr int IS_OP = 76;

        // Load operations
        constexpr int LOAD_CONST = 83;
        constexpr int LOAD_FAST = 85;
        constexpr int LOAD_FAST_LOAD_FAST = 88;
        constexpr int LOAD_FAST_CHECK = 87;
        constexpr int LOAD_FAST_AND_CLEAR = 86;
        constexpr int LOAD_GLOBAL = 91;
        constexpr int LOAD_NAME = 92;
        constexpr int LOAD_ATTR = 82;
        constexpr int LOAD_DEREF = 84;
        constexpr int LOAD_LOCALS = 25;

        // Store operations
        constexpr int STORE_FAST = 110;
        constexpr int STORE_FAST_LOAD_FAST = 111;
        constexpr int STORE_FAST_STORE_FAST = 112;
        constexpr int STORE_GLOBAL = 113;
        constexpr int STORE_NAME = 114;
        constexpr int STORE_ATTR = 108;
        constexpr int STORE_SUBSCR = 39;
        constexpr int STORE_SLICE = 38;
        constexpr int STORE_DEREF = 109;

        // Delete operations
        constexpr int DELETE_FAST = 65;
        constexpr int DELETE_GLOBAL = 66;
        constexpr int DELETE_NAME = 67;
        constexpr int DELETE_ATTR = 63;
        constexpr int DELETE_SUBSCR = 9;
        constexpr int DELETE_DEREF = 64;

        // Jump operations
        constexpr int JUMP_FORWARD = 79;
        constexpr int JUMP_BACKWARD = 77;
        constexpr int JUMP_BACKWARD_NO_INTERRUPT = 78;
        constexpr int POP_JUMP_IF_FALSE = 97;
        constexpr int POP_JUMP_IF_TRUE = 100;
        constexpr int POP_JUMP_IF_NONE = 98;
        constexpr int POP_JUMP_IF_NOT_NONE = 99;

        // Iteration
        constexpr int GET_ITER = 19;
        constexpr int FOR_ITER = 72;
        constexpr int END_FOR = 11;

        // Build operations
        constexpr int BUILD_LIST = 47;
        constexpr int BUILD_TUPLE = 52;
        constexpr int BUILD_MAP = 48;
        constexpr int BUILD_SET = 49;
        constexpr int BUILD_STRING = 51;
        constexpr int BUILD_SLICE = 50;
        constexpr int BUILD_CONST_KEY_MAP = 46;

        // List/Set/Dict operations
        constexpr int LIST_APPEND = 80;
        constexpr int LIST_EXTEND = 81;
        constexpr int SET_ADD = 105;
        constexpr int SET_UPDATE = 107;
        constexpr int MAP_ADD = 95;
        constexpr int DICT_MERGE = 68;
        constexpr int DICT_UPDATE = 69;

        // Unpack operations
        constexpr int UNPACK_SEQUENCE = 117;
        constexpr int UNPACK_EX = 116;

        // Call operations
        constexpr int CALL = 53;
        constexpr int CALL_KW = 57;
        constexpr int CALL_FUNCTION_EX = 54;
        constexpr int CALL_INTRINSIC_1 = 55;
        constexpr int CALL_INTRINSIC_2 = 56;

        // Return operations
        constexpr int RETURN_VALUE = 36;
        constexpr int RETURN_CONST = 103;
        constexpr int RETURN_GENERATOR = 35;

        // Function/Class
        constexpr int MAKE_FUNCTION = 26;
        constexpr int MAKE_CELL = 94;
        constexpr int COPY_FREE_VARS = 62;
        constexpr int SET_FUNCTION_ATTRIBUTE = 106;
        constexpr int LOAD_BUILD_CLASS = 24;

        // Import
        constexpr int IMPORT_NAME = 75;
        constexpr int IMPORT_FROM = 74;

        // Exception handling
        constexpr int PUSH_EXC_INFO = 33;
        constexpr int POP_EXCEPT = 31;
        constexpr int RAISE_VARARGS = 101;
        constexpr int RERAISE = 102;
        constexpr int CHECK_EXC_MATCH = 7;
        constexpr int CHECK_EG_MATCH = 6;
        constexpr int CLEANUP_THROW = 8;

        // With/Async
        constexpr int BEFORE_WITH = 2;
        constexpr int BEFORE_ASYNC_WITH = 1;
        constexpr int WITH_EXCEPT_START = 44;
        constexpr int GET_AWAITABLE = 73;
        constexpr int GET_AITER = 16;
        constexpr int GET_ANEXT = 18;
        constexpr int END_ASYNC_FOR = 10;
        constexpr int END_SEND = 12;
        constexpr int SEND = 104;
        constexpr int YIELD_VALUE = 118;
        constexpr int GET_YIELD_FROM_ITER = 21;

        // Match
        constexpr int MATCH_MAPPING = 28;
        constexpr int MATCH_SEQUENCE = 29;
        constexpr int MATCH_KEYS = 27;
        constexpr int MATCH_CLASS = 96;

        // Format
        constexpr int FORMAT_SIMPLE = 14;
        constexpr int FORMAT_WITH_SPEC = 15;
        constexpr int CONVERT_VALUE = 60;

        // Misc
        constexpr int RESUME = 149;
        constexpr int EXTENDED_ARG = 71;
        constexpr int GET_LEN = 20;
        constexpr int LOAD_ASSERTION_ERROR = 23;
        constexpr int SETUP_ANNOTATIONS = 37;
        constexpr int EXIT_INIT_CHECK = 13;
        constexpr int INTERPRETER_EXIT = 22;
        constexpr int ENTER_EXECUTOR = 70;
        constexpr int LOAD_SUPER_ATTR = 93;

        // Extended opcodes (256+)
        constexpr int JUMP = 256;
        constexpr int JUMP_NO_INTERRUPT = 257;
        constexpr int LOAD_CLOSURE = 258;
        constexpr int LOAD_METHOD = 259;
        constexpr int LOAD_SUPER_METHOD = 260;
        constexpr int LOAD_ZERO_SUPER_ATTR = 261;
        constexpr int LOAD_ZERO_SUPER_METHOD = 262;
        constexpr int POP_BLOCK = 263;
        constexpr int SETUP_CLEANUP = 264;
        constexpr int SETUP_FINALLY = 265;
        constexpr int SETUP_WITH = 266;
        constexpr int STORE_FAST_MAYBE_NULL = 267;

        // BINARY_OP argument values
        namespace binop
        {
            constexpr int ADD = 0;
            constexpr int AND = 1;
            constexpr int FLOOR_DIVIDE = 2;
            constexpr int LSHIFT = 3;
            constexpr int MATRIX_MULTIPLY = 4;
            constexpr int MULTIPLY = 5;
            constexpr int REMAINDER = 6;
            constexpr int OR = 7;
            constexpr int POWER = 8;
            constexpr int RSHIFT = 9;
            constexpr int SUBTRACT = 10;
            constexpr int TRUE_DIVIDE = 11;
            constexpr int XOR = 12;
            // Inplace variants
            constexpr int INPLACE_ADD = 13;
            constexpr int INPLACE_AND = 14;
            constexpr int INPLACE_FLOOR_DIVIDE = 15;
            constexpr int INPLACE_LSHIFT = 16;
            constexpr int INPLACE_MATRIX_MULTIPLY = 17;
            constexpr int INPLACE_MULTIPLY = 18;
            constexpr int INPLACE_REMAINDER = 19;
            constexpr int INPLACE_OR = 20;
            constexpr int INPLACE_POWER = 21;
            constexpr int INPLACE_RSHIFT = 22;
            constexpr int INPLACE_SUBTRACT = 23;
            constexpr int INPLACE_TRUE_DIVIDE = 24;
            constexpr int INPLACE_XOR = 25;
        }

        // COMPARE_OP values (extracted via arg >> 5)
        namespace cmpop
        {
            constexpr int LT = 0; // <
            constexpr int LE = 1; // <=
            constexpr int EQ = 2; // ==
            constexpr int NE = 3; // !=
            constexpr int GT = 4; // >
            constexpr int GE = 5; // >=
        }

    } // namespace op
} // namespace justjit
