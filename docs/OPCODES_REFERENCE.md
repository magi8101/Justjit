# Python 3.13 Bytecode Opcodes Reference

This document provides a comprehensive reference for Python 3.13 bytecode opcodes, categorizing them into **implemented** and **missing** opcodes for the JustJIT compiler.

**Source:** [Python 3.13 dis module documentation](https://docs.python.org/3.13/library/dis.html)

---

## Table of Contents

1. [Implemented Opcodes (11)](#implemented-opcodes)
2. [Missing Opcodes (64)](#missing-opcodes)
   - [Control Flow (6)](#control-flow)
   - [Comparisons (3)](#comparisons)
   - [Stack Manipulation (4)](#stack-manipulation)
   - [Collections (8)](#collections)
   - [Unpacking (2)](#unpacking)
   - [Functions & Closures (5)](#functions--closures)
   - [Method Calls (3)](#method-calls)
   - [Iteration (3)](#iteration)
   - [Attribute/Global Operations (7)](#attributeglobal-operations)
   - [Subscript Operations (2)](#subscript-operations)
   - [Unary Operations (4)](#unary-operations)
   - [Imports (3)](#imports)
   - [Exception Handling (5)](#exception-handling)
   - [Advanced Features (9)](#advanced-features)
3. [Implementation Details](#implementation-details)

---

## Implemented Opcodes

These 11 opcodes are currently implemented in JustJIT:

### **1. POP_TOP** (Opcode: 1)
```python
STACK.pop()
```
**Description:** Removes the top-of-stack item.

**Implementation:** Basic stack pop operation.

---

### **60. STORE_SUBSCR** (Opcode: 60)
```python
value = STACK.pop()       # TOS
container = STACK.pop()   # TOS1
key = STACK.pop()         # TOS2
container[key] = value
```
**Description:** Implements subscript assignment `container[key] = value`.

**Implementation:** Uses PyObject_SetItem via Python C API.

**Stack order:** Pop value first (TOS), then container (TOS1), then key (TOS2).

---

### **83. RETURN_VALUE** (Opcode: 83)
```python
return STACK[-1]
```
**Description:** Returns with `STACK[-1]` to the caller of the function.

**Implementation:** Returns the top of stack as function result.

---

### **100. LOAD_CONST** (Opcode: 100)
```python
STACK.append(co_consts[consti])
```
**Description:** Pushes `co_consts[consti]` onto the stack.

**Implementation:** Loads constants from code object's constant pool.

---

### **106. LOAD_ATTR** (Opcode: 106)
```python
STACK[-1] = getattr(STACK[-1], co_names[namei>>1])
```
**Description:** Loads an attribute from an object.

**Implementation:** Uses PyObject_GetAttr via Python C API.

**Note:** If the low bit of `namei` is set, this attempts to load a method and pushes `NULL` and the method/attribute.

---

### **116. LOAD_GLOBAL** (Opcode: 116)
```python
# Loads global variable
STACK.append(globals()[co_names[namei>>1]])
# If low bit is set, also pushes NULL
if namei & 1:
    STACK.append(NULL)
```
**Description:** Loads the global named `co_names[namei>>1]` onto the stack.

**Implementation:** Looks up global objects from pre-compiled global table. If low bit is set, pushes `NULL` after the global (for method calling convention).

**Python 3.13 Change:** Low bit controls NULL push for calling convention.

---

### **122. BINARY_OP** (Opcode: 122)
```python
rhs = STACK.pop()
lhs = STACK.pop()
STACK.append(lhs op rhs)
```
**Description:** Implements binary and in-place operators.

**Supported operations:**
- `0`: ADD (`+`)
- `1`: AND (`&`)
- `2`: FLOOR_DIVIDE (`//`)
- `3`: LSHIFT (`<<`)
- `4`: MATRIX_MULTIPLY (`@`)
- `5`: MULTIPLY (`*`)
- `6`: REMAINDER (`%`)
- `7`: OR (`|`)
- `8`: POWER (`**`)
- `9`: RSHIFT (`>>`)
- `10`: SUBTRACT (`-`)
- `11`: TRUE_DIVIDE (`/`)
- `12`: XOR (`^`)
- `13`: INPLACE_ADD (`+=`)

**Implementation:** Direct LLVM IR generation for integer operations, PyObject C API for generic objects.

---

### **124. LOAD_FAST** (Opcode: 124)
```python
STACK.append(co_varnames[var_num])
```
**Description:** Pushes a reference to the local `co_varnames[var_num]` onto the stack.

**Implementation:** Loads from local variable array.

**Python 3.12 Change:** Only used when local is guaranteed to be initialized (cannot raise UnboundLocalError).

---

### **125. STORE_FAST** (Opcode: 125)
```python
co_varnames[var_num] = STACK.pop()
```
**Description:** Stores `STACK.pop()` into the local `co_varnames[var_num]`.

**Implementation:** Stores to local variable array.

---

### **149. RESUME** (Opcode: 149)
```python
# No-op for basic execution
```
**Description:** A no-op that performs internal tracing, debugging and optimization checks.

**Implementation:** Currently implemented as no-op.

**Context bits:**
- `0`: Start of function (not generator/coroutine/async generator)
- `1`: After `yield` expression
- `2`: After `yield from` expression
- `3`: After `await` expression

---

### **171. CALL** (Opcode: 171)
```python
# Stack layout: [callable, self_or_null, arg1, arg2, ...]
argc = oparg  # excludes self_or_null
callable = STACK[-2-argc]
self_or_null = STACK[-1-argc]
args = STACK[-argc:]
result = callable(*args)
```
**Description:** Calls a callable object with positional arguments.

**Implementation:** Uses PyObject_Call via Python C API. Stack is accessed by indices (not sequential popping) to match CPython's exact behavior.

**Python 3.13 Changes:**
- Callable now always at same position on stack
- `argc` excludes `self` parameter
- Keyword arguments handled by CALL_KW (separate opcode)

**Critical Implementation Detail:** Must use stack indices like `STACK[-2-oparg]`, not sequential popping, to match CPython's implementation.

---

## Missing Opcodes

### Control Flow

#### **110. JUMP_FORWARD** (delta)
```python
# Increment bytecode counter by delta
```
**Description:** Increments bytecode counter by delta.

**Use Case:** Essential for if/else branches and control flow.

**Priority:** 🔥 HIGH - Required for basic control flow.

---

#### **114. POP_JUMP_IF_FALSE** (delta)
```python
if not STACK[-1]:  # STACK[-1] must be bool
    bytecode_counter += delta
STACK.pop()
```
**Description:** If `STACK[-1]` is false, increments the bytecode counter by delta. `STACK[-1]` is popped.

**Use Case:** Essential for if statements and while loops.

**Priority:** 🔥 HIGH - Required for conditionals.

**Python 3.13 Change:** Now requires exact `bool` operand (not truthy/falsy).

---

#### **115. POP_JUMP_IF_TRUE** (delta)
```python
if STACK[-1]:  # STACK[-1] must be bool
    bytecode_counter += delta
STACK.pop()
```
**Description:** If `STACK[-1]` is true, increments the bytecode counter by delta. `STACK[-1]` is popped.

**Use Case:** Alternative conditional jumps.

**Priority:** 🔥 HIGH - Required for conditionals.

**Python 3.13 Change:** Now requires exact `bool` operand.

---

#### **126. POP_JUMP_IF_NONE** (delta)
```python
if STACK[-1] is None:
    bytecode_counter += delta
STACK.pop()
```
**Description:** If `STACK[-1]` is `None`, increments the bytecode counter by delta.

**Use Case:** None checking in conditionals.

**Priority:** Medium

---

#### **127. POP_JUMP_IF_NOT_NONE** (delta)
```python
if STACK[-1] is not None:
    bytecode_counter += delta
STACK.pop()
```
**Description:** If `STACK[-1]` is not `None`, increments the bytecode counter by delta.

**Use Case:** None checking in conditionals.

**Priority:** Medium

---

#### **140. JUMP_BACKWARD** (delta)
```python
# Decrement bytecode counter by delta
# Check for interrupts (KeyboardInterrupt, etc.)
```
**Description:** Decrements bytecode counter by delta. Checks for interrupts.

**Use Case:** Essential for loops (while, for).

**Priority:** 🔥 HIGH - Required for loops.

---

### Comparisons

#### **107. COMPARE_OP** (opname)
```python
rhs = STACK.pop()
lhs = STACK.pop()
result = lhs <opname> rhs  # <, >, ==, !=, <=, >=
if opname & 16:  # Force bool conversion
    result = bool(result)
STACK.append(result)
```
**Description:** Performs a Boolean comparison operation.

**Operation names** (from `cmp_op[opname >> 5]`):
- `<` (less than)
- `<=` (less or equal)
- `==` (equal)
- `!=` (not equal)
- `>` (greater than)
- `>=` (greater or equal)

**Use Case:** Essential for if/while conditions.

**Priority:** 🔥 HIGH - Required for conditionals.

**Python 3.13 Change:** Fifth-lowest bit indicates forced conversion to `bool`.

---

#### **117. IS_OP** (invert)
```python
rhs = STACK.pop()
lhs = STACK.pop()
if invert:
    result = lhs is not rhs
else:
    result = lhs is rhs
STACK.append(result)
```
**Description:** Performs `is` comparison, or `is not` if `invert` is 1.

**Use Case:** Identity comparison (commonly `if x is None:`).

**Priority:** Medium

---

#### **118. CONTAINS_OP** (invert)
```python
item = STACK.pop()
container = STACK.pop()
if invert:
    result = item not in container
else:
    result = item in container
STACK.append(result)
```
**Description:** Performs `in` comparison, or `not in` if `invert` is 1.

**Use Case:** Membership testing (`if x in list:`).

**Priority:** Medium

---

### Stack Manipulation

#### **2. ROT_TWO**
```python
STACK[-1], STACK[-2] = STACK[-2], STACK[-1]
```
**Description:** Swaps the top two stack items.

**Use Case:** Stack manipulation for complex expressions.

**Priority:** Low

---

#### **3. ROT_THREE**
```python
STACK[-1], STACK[-2], STACK[-3] = STACK[-3], STACK[-1], STACK[-2]
```
**Description:** Rotates the top three stack items.

**Priority:** Low

---

#### **4. DUP_TOP**
```python
STACK.append(STACK[-1])
```
**Description:** Duplicates the top of stack.

**Use Case:** When value is needed multiple times.

**Priority:** Medium

---

#### **5. ROT_FOUR**
```python
STACK[-1], STACK[-2], STACK[-3], STACK[-4] = STACK[-4], STACK[-1], STACK[-2], STACK[-3]
```
**Description:** Rotates the top four stack items.

**Priority:** Low

---

### Collections

#### **102. BUILD_TUPLE** (count)
```python
if count == 0:
    value = ()
else:
    value = tuple(STACK[-count:])
    STACK = STACK[:-count]
STACK.append(value)
```
**Description:** Creates a tuple consuming `count` items from the stack.

**Use Case:** Common in function calls and unpacking.

**Priority:** 🔥 HIGH - Very common operation.

---

#### **103. BUILD_LIST** (count)
```python
if count == 0:
    value = []
else:
    value = list(STACK[-count:])
    STACK = STACK[:-count]
STACK.append(value)
```
**Description:** Creates a list consuming `count` items from the stack.

**Use Case:** Very common data structure.

**Priority:** 🔥 HIGH - Essential for list literals.

---

#### **104. BUILD_SET** (count)
```python
if count == 0:
    value = set()
else:
    value = set(STACK[-count:])
    STACK = STACK[:-count]
STACK.append(value)
```
**Description:** Creates a set consuming `count` items from the stack.

**Use Case:** Set literals `{1, 2, 3}`.

**Priority:** Medium

---

#### **105. BUILD_MAP** (count)
```python
# Pops 2 * count items
# Creates dict: {..., STACK[-4]: STACK[-3], STACK[-2]: STACK[-1]}
items = []
for i in range(count):
    value = STACK.pop()
    key = STACK.pop()
    items.append((key, value))
STACK.append(dict(items))
```
**Description:** Creates a dictionary from stack items.

**Use Case:** Dictionary literals `{'a': 1, 'b': 2}`.

**Priority:** Medium

---

#### **133. BUILD_SLICE** (argc)
```python
if argc == 2:
    end = STACK.pop()
    start = STACK.pop()
    STACK.append(slice(start, end))
elif argc == 3:
    step = STACK.pop()
    end = STACK.pop()
    start = STACK.pop()
    STACK.append(slice(start, end, step))
```
**Description:** Builds a slice object.

**Use Case:** Slicing operations `list[start:end:step]`.

**Priority:** Medium

---

#### **145. LIST_APPEND** (i)
```python
item = STACK.pop()
list.append(STACK[-i], item)
```
**Description:** Appends to a list at position `-i`.

**Use Case:** List comprehensions `[x for x in range(10)]`.

**Priority:** Medium

---

#### **146. SET_ADD** (i)
```python
item = STACK.pop()
set.add(STACK[-i], item)
```
**Description:** Adds to a set at position `-i`.

**Use Case:** Set comprehensions `{x for x in range(10)}`.

**Priority:** Low

---

#### **147. MAP_ADD** (i)
```python
value = STACK.pop()
key = STACK.pop()
dict.__setitem__(STACK[-i], key, value)
```
**Description:** Adds key-value pair to dict at position `-i`.

**Use Case:** Dict comprehensions `{k: v for k, v in items}`.

**Priority:** Low

---

### Unpacking

#### **92. UNPACK_SEQUENCE** (count)
```python
assert len(STACK[-1]) == count
STACK.extend(STACK.pop()[:-count-1:-1])
```
**Description:** Unpacks `STACK[-1]` into `count` individual values, put onto stack right-to-left.

**Use Case:** Tuple/list unpacking `a, b = [1, 2]`.

**Priority:** 🔥 HIGH - Very common pattern.

---

#### **94. UNPACK_EX** (counts)
```python
# Unpacks with starred target: a, *b, c = iterable
# Low byte: values before list
# High byte: values after list
```
**Description:** Unpacks an iterable with starred target.

**Use Case:** Extended unpacking `a, *rest, b = items`.

**Priority:** Medium

---

### Functions & Closures

#### **132. MAKE_FUNCTION**
```python
# Creates function object from code object at STACK[-1]
code = STACK.pop()
func = FunctionType(code, globals())
STACK.append(func)
```
**Description:** Pushes a new function object built from the code object.

**Use Case:** Defining nested functions.

**Priority:** Medium

---

#### **135. LOAD_CLOSURE** (i)
```python
# Pushes cell from slot i (free variable)
STACK.append(cells[i])
```
**Description:** Loads a closure variable cell.

**Use Case:** Closures and nested functions.

**Priority:** Medium

---

#### **136. LOAD_DEREF** (i)
```python
# Loads value from cell in slot i
STACK.append(cells[i].cell_contents)
```
**Description:** Loads the cell contained in slot `i` of the "fast locals" storage.

**Use Case:** Accessing closure variables.

**Priority:** Medium

---

#### **137. STORE_DEREF** (i)
```python
# Stores to cell in slot i
cells[i].cell_contents = STACK.pop()
```
**Description:** Stores into closure variable cell.

**Use Case:** Modifying closure variables.

**Priority:** Medium

---

#### **138. DELETE_DEREF** (i)
```python
# Empties cell in slot i
del cells[i].cell_contents
```
**Description:** Empties the cell contained in slot `i`.

**Use Case:** Deleting closure variables.

**Priority:** Low

---

### Method Calls

#### **160. LOAD_METHOD**
```python
# Optimized unbound method lookup
# Now emitted as LOAD_ATTR with flag set
```
**Description:** Optimized unbound method lookup (pseudo-instruction, replaced by LOAD_ATTR).

**Priority:** Low (handled by LOAD_ATTR)

---

#### **161. CALL_METHOD**
```python
# Calls method loaded by LOAD_METHOD
# Replaced by CALL in Python 3.11+
```
**Description:** Calls method (pseudo-instruction).

**Priority:** Low (handled by CALL)

---

#### **142. CALL_FUNCTION_EX** (flags)
```python
# Calls callable with *args and **kwargs unpacking
if flags & 0x01:
    kwargs = STACK.pop()  # mapping
args = STACK.pop()  # iterable
callable = STACK.pop()
result = callable(*args, **kwargs if flags else {})
STACK.append(result)
```
**Description:** Calls a callable with variable positional/keyword arguments.

**Use Case:** `func(*args, **kwargs)` calls.

**Priority:** Medium

---

### Iteration

#### **68. GET_ITER**
```python
STACK[-1] = iter(STACK[-1])
```
**Description:** Implements `STACK[-1] = iter(STACK[-1])`.

**Use Case:** Essential for for-loops.

**Priority:** 🔥 HIGH - Required for iteration.

---

#### **93. FOR_ITER** (delta)
```python
iterator = STACK[-1]
try:
    value = next(iterator)
    STACK.append(value)
except StopIteration:
    STACK.pop()  # Remove iterator
    bytecode_counter += delta
```
**Description:** Calls `__next__()` on iterator. If exhausted, jumps by delta.

**Use Case:** Essential for for-loops.

**Priority:** 🔥 HIGH - Required for iteration.

**Python 3.12 Change:** Iterator no longer popped when exhausted (stays on stack until loop end).

---

#### **134. JUMP_BACKWARD_NO_INTERRUPT** (delta)
```python
# Decrement bytecode counter by delta
# Does NOT check for interrupts
```
**Description:** Decrements bytecode counter without checking for interrupts.

**Use Case:** Optimization for tight loops.

**Priority:** Low

---

### Attribute/Global Operations

#### **95. STORE_ATTR** (namei)
```python
obj = STACK.pop()
value = STACK.pop()
obj.name = value
```
**Description:** Implements `obj.name = value`.

**Use Case:** Attribute assignment.

**Priority:** Medium

---

#### **96. DELETE_ATTR** (namei)
```python
obj = STACK.pop()
del obj.name
```
**Description:** Implements `del obj.name`.

**Use Case:** Attribute deletion.

**Priority:** Low

---

#### **97. STORE_GLOBAL** (namei)
```python
# Works as STORE_NAME, but stores globally
globals()[co_names[namei]] = STACK.pop()
```
**Description:** Stores a global variable.

**Use Case:** Global variable assignment.

**Priority:** Medium

---

#### **98. DELETE_GLOBAL** (namei)
```python
del globals()[co_names[namei]]
```
**Description:** Deletes a global variable.

**Use Case:** Global variable deletion.

**Priority:** Low

---

#### **101. LOAD_NAME** (namei)
```python
# Lookup order: locals -> globals -> builtins
name = co_names[namei]
value = locals().get(name) or globals().get(name) or builtins[name]
STACK.append(value)
```
**Description:** Loads a name from locals, globals, or builtins.

**Use Case:** Name resolution in module/class scope.

**Priority:** Medium

---

#### **90. STORE_NAME** (namei)
```python
name = co_names[namei]
locals()[name] = STACK.pop()
```
**Description:** Implements `name = STACK.pop()`.

**Use Case:** Name assignment in module/class scope.

**Priority:** Medium

---

#### **91. DELETE_NAME** (namei)
```python
del locals()[co_names[namei]]
```
**Description:** Implements `del name`.

**Use Case:** Name deletion.

**Priority:** Low

---

### Subscript Operations

#### **25. BINARY_SUBSCR**
```python
key = STACK.pop()
container = STACK.pop()
STACK.append(container[key])
```
**Description:** Implements `container[key]` (read access).

**Use Case:** List/dict access `x = list[i]`.

**Priority:** 📊 MEDIUM-HIGH - Very common operation.

---

#### **61. DELETE_SUBSCR**
```python
key = STACK.pop()         # TOS
container = STACK.pop()   # TOS1
del container[key]
```
**Description:** Implements `del container[key]`.

**Use Case:** Deleting from containers.

**Priority:** Low

---

### Unary Operations

#### **10. UNARY_POSITIVE**
```python
STACK[-1] = +STACK[-1]
```
**Description:** Implements unary `+`.

**Use Case:** Unary plus operator.

**Priority:** Low

---

#### **11. UNARY_NEGATIVE**
```python
STACK[-1] = -STACK[-1]
```
**Description:** Implements unary `-`.

**Use Case:** Negation operator `-x`.

**Priority:** Medium

---

#### **12. UNARY_NOT**
```python
STACK[-1] = not STACK[-1]
```
**Description:** Implements `not` operator.

**Use Case:** Boolean negation.

**Priority:** Medium

**Python 3.13 Change:** Now requires exact `bool` operand.

---

#### **15. UNARY_INVERT**
```python
STACK[-1] = ~STACK[-1]
```
**Description:** Implements bitwise NOT `~`.

**Use Case:** Bitwise inversion.

**Priority:** Low

---

### Imports

#### **108. IMPORT_NAME** (namei)
```python
fromlist = STACK.pop()  # STACK[-1]
level = STACK.pop()      # STACK[-2]
module = __import__(co_names[namei], globals(), locals(), fromlist, level)
STACK.append(module)
```
**Description:** Imports a module.

**Use Case:** `import` statements.

**Priority:** Medium

---

#### **109. IMPORT_FROM** (namei)
```python
module = STACK[-1]
attr = getattr(module, co_names[namei])
STACK.append(attr)
```
**Description:** Loads attribute from module.

**Use Case:** `from module import name`.

**Priority:** Medium

---

#### **84. IMPORT_STAR**
```python
# Performs import * for the named module
module = STACK.pop()
# Import all public names into namespace
```
**Description:** Implements `from module import *`.

**Use Case:** Star imports.

**Priority:** Low

---

### Exception Handling

#### **130. RAISE_VARARGS** (argc)
```python
if argc == 0:
    raise  # Re-raise
elif argc == 1:
    raise STACK[-1]
elif argc == 2:
    raise STACK[-2] from STACK[-1]
```
**Description:** Raises an exception.

**Use Case:** `raise` statements.

**Priority:** Medium

---

#### **119. RERAISE**
```python
# Re-raises exception on top of stack
exception = STACK[-1]
raise exception
```
**Description:** Re-raises the exception currently on top of the stack.

**Use Case:** Exception handling.

**Priority:** Medium

---

#### **35. PUSH_EXC_INFO**
```python
# Pushes current exception to top of stack
value = STACK.pop()
STACK.append(current_exception)
STACK.append(value)
```
**Description:** Pops a value, pushes current exception, then pushes value back.

**Use Case:** Exception handlers.

**Priority:** Medium

---

#### **89. POP_EXCEPT**
```python
# Pops exception state from stack
exception_state = STACK.pop()
# Restore exception state
```
**Description:** Pops a value used to restore exception state.

**Use Case:** Exiting exception handlers.

**Priority:** Medium

---

#### **36. CHECK_EXC_MATCH**
```python
exc_type = STACK.pop()  # STACK[-1]
exc = STACK[-1]         # STACK[-2]
result = isinstance(exc, exc_type)
STACK.append(result)
```
**Description:** Tests whether `STACK[-2]` is an exception matching `STACK[-1]`.

**Use Case:** `except ExceptionType:` matching.

**Priority:** Medium

---

### Advanced Features

#### **143. SETUP_WITH** (target)
```python
# Pseudo-instruction for with statement setup
# Replaced before bytecode generation
```
**Description:** Set up exception handler for with block.

**Priority:** Low (pseudo-instruction)

---

#### **49. WITH_EXCEPT_START**
```python
# Calls __exit__ function with exception info
exit_func = STACK[-4]
result = exit_func(type, val, tb)
STACK.append(result)
```
**Description:** Calls context manager's `__exit__` when exception occurs.

**Use Case:** `with` statement exception handling.

**Priority:** Low

---

#### **70. PRINT_EXPR**
```python
# Prints expression in REPL
value = STACK.pop()
print(repr(value))
```
**Description:** Prints the argument to stdout (used in REPL).

**Priority:** Low

---

#### **71. LOAD_BUILD_CLASS**
```python
STACK.append(builtins.__build_class__)
```
**Description:** Pushes `builtins.__build_class__()` onto stack.

**Use Case:** Class definition.

**Priority:** Low

---

#### **86. YIELD_VALUE**
```python
value = STACK.pop()
# Yield value from generator
```
**Description:** Yields value from a generator.

**Use Case:** Generator functions.

**Priority:** Low

---

#### **75. RETURN_GENERATOR**
```python
# Create generator from current frame
# Return the generator
```
**Description:** Creates a generator/coroutine/async generator from current frame.

**Use Case:** Generator initialization.

**Priority:** Low

---

#### **73. GET_AWAITABLE** (where)
```python
o = STACK[-1]
if is_coroutine(o) or is_generator_with_CO_ITERABLE_COROUTINE(o):
    awaitable = o
else:
    awaitable = o.__await__()
STACK[-1] = awaitable
```
**Description:** Gets awaitable object for `await` expression.

**Use Case:** Async/await.

**Priority:** Low

---

#### **50. GET_AITER**
```python
STACK[-1] = STACK[-1].__aiter__()
```
**Description:** Implements `STACK[-1] = STACK[-1].__aiter__()`.

**Use Case:** Async iteration.

**Priority:** Low

---

#### **51. GET_ANEXT**
```python
anext = STACK[-1].__anext__()
awaitable = get_awaitable(anext)
STACK.append(awaitable)
```
**Description:** Gets next item from async iterator.

**Use Case:** Async for loops.

**Priority:** Low

---

## Implementation Details

### Current JustJIT Implementation Strategy

**Architecture:** Direct bytecode interpreter in LLVM
- Simulates Python's stack machine in LLVM IR
- Uses Python C API for PyObject operations
- Generates native code via LLVM backend

**Stack Model:**
```cpp
std::vector<llvm::Value*> stack;
// Stack grows upward: higher indices = top of stack
// STACK[-1] is stack.back()
// STACK[-2] is stack[stack.size()-2]
```

**Critical Implementation Notes:**

1. **Stack Indexing (CALL opcode):**
   ```cpp
   // CPython uses indices, NOT sequential popping:
   size_t base = stack.size() - num_args - 2;
   llvm::Value* callable = stack[base];           // [-2-oparg]
   llvm::Value* self_or_null = stack[base + 1];   // [-1-oparg]
   // args at stack[base + 2] onwards
   ```

2. **LOAD_GLOBAL Push Order:**
   ```cpp
   // Push global FIRST, then NULL (if flag set)
   stack.push_back(global_obj);
   if (push_null) {
       stack.push_back(null_ptr);
   }
   // Creates: [global, NULL] for CALL to use as [callable, self_or_null]
   ```

3. **Python 3.13 Changes:**
   - Jump targets are relative offsets (not absolute)
   - CALL opcode changed: `argc` excludes self parameter
   - Keyword calls use separate CALL_KW opcode
   - Some opcodes now require exact `bool` operands (not truthy/falsy)

### Recommended Implementation Priority

**Phase 1: Control Flow (Enable conditionals and loops)**
1. COMPARE_OP - Comparisons
2. POP_JUMP_IF_FALSE - If statements
3. POP_JUMP_IF_TRUE - Alternative conditionals
4. JUMP_FORWARD - Forward branches
5. JUMP_BACKWARD - Loops
6. GET_ITER - Start iteration
7. FOR_ITER - Loop iteration

**Phase 2: Data Structures**
1. BUILD_TUPLE - Tuple creation
2. BUILD_LIST - List creation
3. UNPACK_SEQUENCE - Unpacking
4. BINARY_SUBSCR - Read access

**Phase 3: Extended Functionality**
1. BUILD_MAP - Dictionaries
2. IS_OP, CONTAINS_OP - Additional comparisons
3. MAKE_FUNCTION - Nested functions
4. Unary operators

### Testing Strategy

Each opcode should be tested with:
1. Basic functionality test
2. Edge cases (empty, single element, large)
3. Type mixing (int, PyObject)
4. Integration with other opcodes

### Reference Implementation

**CPython source:** `Python/generated_cases.c.h` (Python 3.13)
- Contains exact stack manipulation for each opcode
- Authoritative reference for implementation details

**Documentation:** `https://docs.python.org/3.13/library/dis.html`

---

## Summary Statistics

- **Total implemented:** 11 opcodes
- **Total missing (common):** 64 opcodes
- **High priority missing:** 10 opcodes
- **Coverage:** ~15% of commonly-used opcodes

**Current capabilities:**
- Basic arithmetic
- Variable load/store
- Function calls
- Attribute access
- Container assignment
- Control flow (if/while/for)
- Comparisons
- Data structure creation
- Unpacking

**Next milestone:** Implement Phase 1 (control flow) to enable basic if/for/while statements.

---

*Last updated: November 29, 2025*
*Python version: 3.13*
*JustJIT version: 0.1.0*
