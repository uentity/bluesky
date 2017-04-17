/// @file
/// @author uentity
/// @date 25.03.2017
/// @brief Python subsystem of BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>

namespace blue_sky { namespace python {
namespace py = pybind11;

void py_bind_common(py::module& m);
void py_bind_messaging(py::module& m);
void py_bind_objbase(py::module& m);

BS_INIT_PY(bs) {
	py_bind_common(m);
	py_bind_objbase(m);
	py_bind_messaging(m);
}

}}
