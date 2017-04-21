// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#include <bs/common.h>
#include <bs/kernel.h>
#include <bs/detail/lib_descriptor.h>
#include <bs/log.h>

#include <pybind11/pybind11.h>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <string>
#include <exception>

//! define TARGET_NAME in compiler equal to output loader's name without extension
#ifndef TARGET_NAME
#define TARGET_NAME bs
#endif

#define S_TARGET_NAME BOOST_PP_STRINGIZE(TARGET_NAME)
#define INIT_FN_NAME BOOST_PP_CAT(init, TARGET_NAME)

using namespace blue_sky;
using namespace std;
namespace py = pybind11;

PYBIND11_PLUGIN(TARGET_NAME) {
	// search for BlueSky's kernel plugin descriptor
	BS_GET_PLUGIN_DESCRIPTOR get_pd_fn;
	if(detail::lib_descriptor::load_sym_glob("bs_get_plugin_descriptor", get_pd_fn) != 0 || !get_pd_fn) {
		BSERROR << log::E("BlueSky kernel descriptor wasn't found or invalid!") << log::end;
		return nullptr;
		//throw std::runtime_error("BlueSky kernel descriptor wasn't found");
	}
	plugin_descriptor* kernel_pd = get_pd_fn();
	// correct global namespace to match TARGET_NAME (otherwise Python throws an error)
	kernel_pd->py_namespace = S_TARGET_NAME;

	auto m = py::module(S_TARGET_NAME, kernel_pd->description.c_str());

	//load plugins with Python subsystem
	give_kernel::Instance().load_plugins(&m);

	return m.ptr();
}

