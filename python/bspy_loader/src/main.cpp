/// @file
/// @author uentity
/// @date 22.11.2018
/// @brief Python loader for blue-sky-re kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/


#include <bs/python/common.h>
#include <bs/log.h>
#include <bs/kernel/errors.h>
#include <bs/kernel/plugins.h>
#include <bs/detail/lib_descriptor.h>

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
	bs_init_py_fn py_init_kernel;
	if(detail::lib_descriptor::load_sym_glob("bs_init_py_subsystem", py_init_kernel) != 0 || !py_init_kernel)
		throw error{"BS kernel", kernel::Error::PythonDisabled};

	// init BS Python subsystem
	py_init_kernel(&m);
	BSOUT << "BlueSky kernel Python subsystem initialized successfully under namespace: {}"
		<< S_TARGET_NAME << bs_end;

	// auto-load plugins
	kernel::plugins::load_plugins();
}

