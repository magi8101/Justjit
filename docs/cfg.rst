Control Flow Graph
==================

JustJIT builds a control flow graph (CFG) from Python bytecode to handle complex control flow. This is essential for proper LLVM IR generation, especially at merge points where PHI nodes are needed.

Why a CFG?
----------

Python bytecode has complex control flow:

- Conditional jumps (``if/else``)
- Loops (``for``, ``while``)
- Exception handlers (``try/except``)
- Pattern matching (``match/case``)

When multiple code paths merge, LLVM's SSA form requires PHI nodes to select between values from different predecessors. The CFG tells us where these merge points are.

Data Structures
---------------

JustJIT uses these data structures (defined in ``jit_core.h``):

BasicBlockInfo
^^^^^^^^^^^^^^

.. code-block:: cpp

   struct BasicBlockInfo {
       int start_offset;          // Bytecode offset where block starts
       int end_offset;            // Bytecode offset where block ends
       std::vector<int> predecessors;    // Blocks that jump here
       std::vector<int> successors;      // Blocks we jump to
       int stack_depth_at_entry;  // Expected stack depth
       bool is_exception_handler; // True if exception handler entry
       bool needs_phi_nodes;      // True if multiple predecessors
       llvm::BasicBlock* llvm_block;     // The LLVM basic block
   };

We use ``std::map<int, BasicBlockInfo>`` keyed by start offset.

**Why std::map?** Maps provide O(log n) lookup by offset and maintain sorted order, which helps when iterating blocks in bytecode order. For typical Python functions (tens to hundreds of instructions), this is efficient enough.

CFGStackState
^^^^^^^^^^^^^

.. code-block:: cpp

   struct CFGStackState {
       std::vector<llvm::Value*> stack;  // Values on stack
       llvm::BasicBlock* from_block;     // Source block
       int from_offset;                  // Source offset
   };

This captures stack state when leaving a block, enabling PHI node creation at merge points.

Three-Phase Analysis
--------------------

CFG construction happens in three phases (lines 1102-1500 in ``jit_core.cpp``):

Phase 1: Find Block Starts
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``find_block_starts()`` function identifies basic block boundaries:

.. code-block:: cpp

   static std::set<int> find_block_starts(
       const std::vector<Instruction>& instructions,
       const std::vector<ExceptionTableEntry>& exception_table
   ) {
       std::set<int> block_starts;
       block_starts.insert(0);  // Entry block always at offset 0

       for (size_t i = 0; i < instructions.size(); ++i) {
           const auto& instr = instructions[i];

           // Conditional jumps create two block starts
           if (instr.opcode == op::POP_JUMP_IF_FALSE ||
               instr.opcode == op::POP_JUMP_IF_TRUE ||
               instr.opcode == op::POP_JUMP_IF_NONE ||
               instr.opcode == op::POP_JUMP_IF_NOT_NONE) {
               block_starts.insert(instr.argval);     // Jump target
               if (i + 1 < instructions.size()) {
                   block_starts.insert(instructions[i+1].offset);  // Fall-through
               }
           }
           // Unconditional jumps
           else if (instr.opcode == op::JUMP_FORWARD ||
                    instr.opcode == op::JUMP_BACKWARD) {
               block_starts.insert(instr.argval);
           }
           // FOR_ITER has two exits
           else if (instr.opcode == op::FOR_ITER) {
               block_starts.insert(instr.argval);     // Exhaustion target
               if (i + 1 < instructions.size()) {
                   block_starts.insert(instructions[i+1].offset);  // Continue
               }
           }
       }

       // Exception handlers are block starts
       for (const auto& exc_entry : exception_table) {
           block_starts.insert(exc_entry.target);
       }

       return block_starts;
   }

**Why std::set?** Sets provide automatic deduplication (important since multiple instructions may target the same offset) and sorted iteration.

Phase 2: Build CFG Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``build_cfg()`` function creates blocks and edges:

.. code-block:: cpp

   static std::map<int, BasicBlockInfo> build_cfg(
       const std::vector<Instruction>& instructions,
       const std::vector<ExceptionTableEntry>& exception_table,
       const std::set<int>& block_starts
   ) {
       std::map<int, BasicBlockInfo> cfg;

       // Create all blocks first
       std::vector<int> sorted_starts(block_starts.begin(), block_starts.end());
       for (size_t b = 0; b < sorted_starts.size(); ++b) {
           int start = sorted_starts[b];
           BasicBlockInfo info;
           info.start_offset = start;
           info.end_offset = (b + 1 < sorted_starts.size()) 
               ? sorted_starts[b + 1]
               : instructions.back().offset + 2;
           info.stack_depth_at_entry = -1;  // Unknown initially
           info.is_exception_handler = false;
           info.needs_phi_nodes = false;
           cfg[start] = info;
       }

       // Mark exception handlers
       for (const auto& exc_entry : exception_table) {
           if (cfg.count(exc_entry.target)) {
               cfg[exc_entry.target].is_exception_handler = true;
               cfg[exc_entry.target].stack_depth_at_entry = exc_entry.depth;
           }
       }

       // Build edges (predecessors/successors)
       // ... analyze each block's terminator instruction

       return cfg;
   }

Phase 3: Compute Stack Depths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``compute_stack_depths()`` function uses forward dataflow analysis:

.. code-block:: cpp

   bool compute_stack_depths(
       std::map<int, BasicBlockInfo>& cfg,
       const std::vector<Instruction>& instructions,
       int initial_stack_depth = 0
   ) {
       // Initialize entry block
       cfg[0].stack_depth_at_entry = initial_stack_depth;

       // Worklist algorithm
       std::queue<int> worklist;
       worklist.push(0);

       while (!worklist.empty()) {
           int offset = worklist.front();
           worklist.pop();

           BasicBlockInfo& block = cfg[offset];
           int depth = block.stack_depth_at_entry;

           // Simulate instructions in this block
           for (each instruction in [block.start, block.end)) {
               depth += stack_effect(instr);
           }

           // Propagate to successors
           for (int succ : block.successors) {
               if (cfg[succ].stack_depth_at_entry == -1) {
                   cfg[succ].stack_depth_at_entry = depth;
                   worklist.push(succ);
               } else if (cfg[succ].stack_depth_at_entry != depth) {
                   // Inconsistent depths - this indicates a bug
                   return false;
               }
           }
       }

       return true;
   }

Exception Handling
------------------

Exception handlers require special CFG treatment:

.. code-block:: cpp

   struct ExceptionTableEntry {
       int32_t start;   // Start of protected range
       int32_t end;     // End of protected range  
       int32_t target;  // Handler entry point
       int32_t depth;   // Stack depth at handler entry
       bool lasti;      // Push last instruction offset?
   };

The exception table is parsed from Python's ``co_exceptiontable`` (line 1536):

.. code-block:: cpp

   for (size_t i = 0; i < py_exception_table.size(); ++i) {
       nb::dict entry_dict = nb::cast<nb::dict>(py_exception_table[i]);
       ExceptionTableEntry entry;
       entry.start = nb::cast<int32_t>(entry_dict["start"]);
       entry.end = nb::cast<int32_t>(entry_dict["end"]);
       entry.target = nb::cast<int32_t>(entry_dict["target"]);
       entry.depth = nb::cast<int32_t>(entry_dict["depth"]);
       entry.lasti = nb::cast<bool>(entry_dict["lasti"]);
       exception_table.push_back(entry);
   }

Pattern Matching
----------------

Python 3.10+ pattern matching creates complex CFG patterns. JustJIT supports these opcodes:

- ``MATCH_SEQUENCE``: Check if subject is a sequence
- ``MATCH_MAPPING``: Check if subject is a mapping
- ``MATCH_CLASS``: Match against class and extract attributes
- ``MATCH_KEYS``: Extract values for specific keys

The ``JITMatchClass`` helper (line 271 in ``jit_core.cpp``) handles class matching:

.. code-block:: cpp

   extern "C" PyObject* JITMatchClass(
       PyObject *subject, PyObject *cls, 
       int nargs, PyObject *names
   ) {
       // Check isinstance
       int is_instance = PyObject_IsInstance(subject, cls);
       if (!is_instance) {
           Py_INCREF(Py_None);
           return Py_None;  // No match
       }

       // Get __match_args__ for positional patterns
       if (nargs > 0) {
           PyObject *match_args = PyObject_GetAttrString(cls, "__match_args__");
           // Extract positional attributes using match_args
       }

       // Extract keyword attributes from 'names' tuple
       // ...

       return attrs;  // Tuple of matched values
   }

PHI Node Generation
-------------------

At blocks with multiple predecessors, JustJIT generates PHI nodes:

.. code-block:: cpp

   // In compile_function, when entering a block
   if (block.predecessors.size() > 1) {
       // Multiple incoming edges - need PHI nodes for stack values
       for (int i = 0; i < stack_depth; i++) {
           llvm::PHINode* phi = builder.CreatePHI(
               ptr_type, block.predecessors.size()
           );
           for (auto& incoming : block_incoming_stacks[offset]) {
               phi->addIncoming(incoming.stack[i], incoming.predecessor);
           }
           stack[i] = phi;
       }
   }

Performance
-----------

CFG analysis is lightweight:

- **Time complexity**: O(n) where n = number of instructions
- **Typical time**: < 5ms for functions up to 500 instructions

This is negligible compared to LLVM optimization and code generation.

Reference Counting in Generators
--------------------------------

Generator stack persistence requires careful reference counting. When saving stack values across yields:

.. code-block:: cpp

   // Before yield: incref stack values being saved
   for (size_t s = 0; s < current_stack_depth; s++) {
       int slot_idx = nlocals + s;
       llvm::Value* slot_ptr = builder.CreateGEP(ptr_type, locals_array,
           llvm::ConstantInt::get(i64_type, slot_idx));
       builder.CreateCall(py_xincref_func, {stack[s]});  // Keep alive
       builder.CreateStore(stack[s], slot_ptr);
   }

On resume, stack values are loaded but not incref'd (they already have refs from save).

**Exception Unwinding:**

When a generator encounters an error (state = -2), remaining stack values must be decref'd to prevent leaks:

.. code-block:: cpp

   // In error block: decref all stack values
   builder.SetInsertPoint(error_block);
   for (size_t s = 0; s < max_stack_depth; s++) {
       int slot_idx = nlocals + s;
       llvm::Value* slot_ptr = builder.CreateGEP(ptr_type, locals_array,
           llvm::ConstantInt::get(i64_type, slot_idx));
       llvm::Value* val = builder.CreateLoad(ptr_type, slot_ptr);
       builder.CreateCall(py_xdecref_func, {val});
       builder.CreateStore(null_ptr, slot_ptr);  // Clear slot
   }
   builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
   builder.CreateRet(null_ptr);

This matches Python's behavior of cleaning up on generator finalization.

