#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "jit_core.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m)
{
     // Disable leak warnings - our extension stores references to globals
     // which creates cycles that Python's GC handles but appear as leaks
     // at interpreter shutdown. See nanobind docs on reference leaks.
     nb::set_leak_warnings(false);

     m.doc() = "Fast Python JIT compiler using LLVM ORC";

     nb::class_<justjit::JITCore>(m, "JIT")
         .def(nb::init<>())
         .def("set_opt_level", &justjit::JITCore::set_opt_level, "level"_a)
         .def("get_opt_level", &justjit::JITCore::get_opt_level)
         .def("set_dump_ir", &justjit::JITCore::set_dump_ir, "dump"_a, "Enable/disable IR capture for debugging")
         .def("get_dump_ir", &justjit::JITCore::get_dump_ir, "Check if IR dump is enabled")
         .def("get_last_ir", &justjit::JITCore::get_last_ir, "Get the LLVM IR from the last compiled function")
         .def("compile", [](justjit::JITCore &self, nb::list instructions, nb::list constants, nb::list names, nb::object globals_dict, nb::object builtins_dict, nb::list closure_cells, nb::list exception_table, const std::string &name, int param_count, int total_locals, int nlocals)
              { return self.compile_function(instructions, constants, names, globals_dict, builtins_dict, closure_cells, exception_table, name, param_count, total_locals, nlocals); }, "instructions"_a, "constants"_a, "names"_a, "globals_dict"_a, "builtins_dict"_a, "closure_cells"_a, "exception_table"_a, "name"_a, "param_count"_a = 2, "total_locals"_a = 3, "nlocals"_a = 3, "Compile a Python function to native code")
         .def("compile_int", [](justjit::JITCore &self, nb::list instructions, nb::list constants, const std::string &name, int param_count, int total_locals)
              { return self.compile_int_function(instructions, constants, name, param_count, total_locals); }, "instructions"_a, "constants"_a, "name"_a, "param_count"_a = 2, "total_locals"_a = 3, "Compile an integer-only function to native code (no Python object overhead)")
         .def("compile_generator", [](justjit::JITCore &self, nb::list instructions, nb::list constants, nb::list names, nb::object globals_dict, nb::object builtins_dict, nb::list closure_cells, nb::list exception_table, const std::string &name, int param_count, int total_locals, int nlocals)
              { return self.compile_generator(instructions, constants, names, globals_dict, builtins_dict, closure_cells, exception_table, name, param_count, total_locals, nlocals); }, "instructions"_a, "constants"_a, "names"_a, "globals_dict"_a, "builtins_dict"_a, "closure_cells"_a, "exception_table"_a, "name"_a, "param_count"_a = 0, "total_locals"_a = 1, "nlocals"_a = 1, "Compile a generator function to a state machine step function")
         .def("lookup", &justjit::JITCore::lookup_symbol, "name"_a)
         .def("get_callable", &justjit::JITCore::get_callable, "name"_a, "param_count"_a)
         .def("get_int_callable", &justjit::JITCore::get_int_callable, "name"_a, "param_count"_a, "Get a callable for an integer-mode function")
         .def("get_generator_callable", &justjit::JITCore::get_generator_callable, "name"_a, "param_count"_a, "total_locals"_a, "func_name"_a, "func_qualname"_a, "Get generator metadata for creating generator objects");

     // Expose the JITGenerator type and creation function
     m.def("create_jit_generator", [](uint64_t step_func_addr, int64_t num_locals, nb::object name, nb::object qualname) {
         auto step_func = reinterpret_cast<justjit::GeneratorStepFunc>(step_func_addr);
         PyObject* gen = justjit::JITGenerator_New(step_func, static_cast<Py_ssize_t>(num_locals),
                                                    name.ptr(), qualname.ptr());
         if (gen == nullptr) {
             throw nb::python_error();
         }
         return nb::steal(gen);
     }, "step_func_addr"_a, "num_locals"_a, "name"_a, "qualname"_a,
        "Create a new JIT generator object from a compiled step function");
     
     // Expose the JITCoroutine type and creation function
     m.def("create_jit_coroutine", [](uint64_t step_func_addr, int64_t num_locals, nb::object name, nb::object qualname) {
         auto step_func = reinterpret_cast<justjit::GeneratorStepFunc>(step_func_addr);
         PyObject* coro = justjit::JITCoroutine_New(step_func, static_cast<Py_ssize_t>(num_locals),
                                                     name.ptr(), qualname.ptr());
         if (coro == nullptr) {
             throw nb::python_error();
         }
         return nb::steal(coro);
     }, "step_func_addr"_a, "num_locals"_a, "name"_a, "qualname"_a,
        "Create a new JIT coroutine object from a compiled step function");
}
