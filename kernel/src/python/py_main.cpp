/// @file
/// @author uentity
/// @date 25.03.2017
/// @brief Python subsystem of BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/python/nparray.h>
#include "../kernel/python_subsyst.h"

namespace blue_sky { namespace python {
namespace py = pybind11;

void py_bind_common(py::module& m);
void py_bind_messaging(py::module& m);
void py_bind_objbase(py::module& m);
void py_bind_misc(py::module& m);
void py_bind_kernel(py::module& m);
void py_bind_tree(py::module& m);
void py_bind_error(py::module& m);
void py_bind_log(py::module& m);

BS_INIT_PY(bs) {
	// save pointer to kernel's Py module
	singleton<kernel::detail::python_subsyst>::Instance().setup_py_kmod(&m);
	// initialize kernel
	kernel::init();
	// shutdown kernel when Py interpreter exists
	Py_AtExit([] { kernel::shutdown(); });

	// invoke bindings
	py_bind_common(m);
	py_bind_log(m);
	py_bind_error(m);
	py_bind_objbase(m);
	py_bind_messaging(m);
	py_bind_misc(m);
	py_bind_kernel(m);
	py_bind_tree(m);
}

}}
