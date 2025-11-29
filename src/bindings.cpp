#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "jit_core.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    m.doc() = "Fast Python JIT compiler using LLVM ORC";
    
    nb::class_<justjit::JITCore>(m, "JIT")
        .def(nb::init<>())
        .def("set_opt_level", &justjit::JITCore::set_opt_level, "level"_a)
        .def("get_opt_level", &justjit::JITCore::get_opt_level)
        .def("compile", 
             [](justjit::JITCore& self, nb::list instructions, nb::list constants, nb::list names, nb::list globals,
                const std::string& name, int param_count, int total_locals) {
                 return self.compile_function(instructions, constants, names, globals, name, param_count, total_locals);
             },
             "instructions"_a, "constants"_a, "names"_a, "globals"_a, "name"_a, "param_count"_a = 2, "total_locals"_a = 3,
             "Compile a Python function to native code")
        .def("lookup", &justjit::JITCore::lookup_symbol, "name"_a)
        .def("get_callable", &justjit::JITCore::get_callable, "name"_a, "param_count"_a);
}
