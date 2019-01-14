/// @file
/// @author uentity
/// @date 22.11.2018
/// @brief Python loader for blue-sky-re kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/


#include <bs/common.h>
#include <bs/error.h>
#include <bs/kernel/misc.h>
#include <bs/kernel/plugins.h>
#include <bs/detail/lib_descriptor.h>
#include <bs/log.h>

#include <pybind11/pybind11.h>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

/// TARGET_NAME must match built library name
#ifndef TARGET_NAME
#define TARGET_NAME bs
#endif

#define S_TARGET_NAME BOOST_PP_STRINGIZE(TARGET_NAME)
#define INIT_FN_NAME BOOST_PP_CAT(init, TARGET_NAME)

using namespace blue_sky;
using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(TARGET_NAME, m) {
	// search for BlueSky's kernel plugin descriptor
	BS_GET_PLUGIN_DESCRIPTOR get_pd_fn;
	if(detail::lib_descriptor::load_sym_glob("bs_get_plugin_descriptor", get_pd_fn) != 0 || !get_pd_fn) {
		throw error("BlueSky kernel descriptor wasn't found or invalid!");
	}
	plugin_descriptor* kernel_pd = get_pd_fn();
	// Python namespace must match TARGET_NAME (otherwise Python throws an error)
	kernel_pd->py_namespace = S_TARGET_NAME;
	m.doc() = kernel_pd->description;

	// initialize kernel
	kernel::init();

	//load plugins with Python subsystem
	kernel::plugins::load_plugins(&m);
}

