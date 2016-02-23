/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_common.h"
#include "bs_import_common.h"
#include "py_bs_exports.h"
#include "py_bs_assert.h"

#include "shared_vector.h"

#include <boost/python/enum.hpp>
#include <boost/python/exception_translator.hpp>

namespace blue_sky { namespace python {

// forward definition of exporting functions
//void py_export_common();
void py_export_typed();
void py_export_combase();
void py_export_kernel();
void py_export_abstract_storage();
void py_export_link();
void py_export_log();
void py_export_messaging();
void py_export_objbase();
void py_export_shell();
void py_export_tree();
void py_export_nparray();


void py_bind_messaging();
void py_bind_objbase();
void py_bind_common();
void py_bind_link();
void py_bind_tree();
void py_bind_kernel();

namespace {
struct deprecated_tag {};
}

BLUE_SKY_INIT_PY_FUN
{
	register_exception_translator<bs_exception>(&exception_translator);

	py_export_vectors ();
	python::py_export_error_codes ();
	python::py_export_assert ();

	// export the rest stuff
	// do it under 'deprecated' namespace
	{
		scope deprecated = class_< deprecated_tag, boost::noncopyable >("deprecated", no_init);
		py_export_messaging();
		py_export_objbase();
		py_export_typed();
		py_export_combase();
		py_export_kernel();
		py_export_abstract_storage();
		py_export_link();
		py_export_log();
		py_export_shell();
		py_export_tree();
	}
	py_export_nparray();

	// new exporting system
	py_bind_common();
	py_bind_messaging();
	py_bind_objbase();
	py_bind_link();
	py_bind_tree();
	py_bind_kernel();
}

}}
