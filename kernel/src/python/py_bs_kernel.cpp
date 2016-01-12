/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_common.h"
#include "bs_exception.h"
#include "bs_kernel_tools.h"

#include "py_bs_kernel.h"
#include "py_bs_log.h"
#include "py_bs_exports.h"

#include <algorithm>
#include <boost/python/overloads.hpp>

using namespace std;
using namespace Loki;
using namespace boost;

namespace blue_sky {
namespace python {

blue_sky::python::py_objbase py_kernel::create_object(const type_descriptor &tt)	{
	return py_objbase(kernel_.create_object(tt));
}

blue_sky::python::py_objbase py_kernel::create_object_by_typename(const char *tname)	{
	return py_objbase(kernel_.create_object(std::string(tname)));
}

std::vector< type_tuple > py_kernel::registered_types()	{
	return kernel_.registered_types();
}

py_kernel::py_kernel()
	: kernel_(blue_sky::give_kernel::Instance())
{}

py_kernel::~py_kernel() {}

blue_sky::kernel::plugins_enum py_kernel::loaded_plugins() {
	return kernel_.loaded_plugins();
}

blue_sky::kernel::types_enum py_kernel::plugin_types(const plugin_descriptor& pd)	{
	return kernel_.plugin_types(pd);
}

blue_sky::kernel::types_enum py_kernel::plugin_types_by_name(const char *pname) {
	return kernel_.plugin_types(std::string(pname));
}

type_tuple py_kernel::find_type(const char *type_str) const {
	return kernel_.find_type(type_str);
}

bool py_kernel::register_type(const plugin_descriptor& pd, const type_descriptor& td) const {
	return kernel_.register_type(pd,td);
}

py_objbase py_kernel::create_object_copy(const py_objbase& src, bool unmanaged) const {
	return py_objbase(kernel_.create_object_copy(src.sp,unmanaged));
}

void py_kernel::free_instance(const py_objbase &obj) const {
	kernel_.free_instance(obj.sp);
}

py_bs_abstract_storage py_kernel::create_storage(const char *filename, const char *format, int flags) const {
	return py_bs_abstract_storage(kernel_.create_storage(filename,format,flags));
}

void py_kernel::close_storage(const py_bs_abstract_storage &storage) const {
	kernel_.close_storage(storage.spstor);
}

error_code py_kernel::load_plugin(const std::string& fname, const std::string version) {
	return kernel_.load_plugin(fname,version);
}

void py_kernel::unload_plugin(const plugin_descriptor& pd) {
	kernel_.unload_plugin(pd);
}

std::string py_kernel::get_last_error() const {
	return kernel_.get_last_error();
}

void py_kernel::add_task(const py_combase& task) {
	kernel_.add_task(task.spcom);
}

py_bs_link py_kernel::bs_root() const {
	return py_bs_link(kernel_.bs_root());
}

//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(kernel_tools_overloads, walk_tree, 1, 1);


void py_export_kernel() {
	class_<py_kernel>("kernel")
		.def("create_object",&py_kernel::create_object)
		.def("create_object",&py_kernel::create_object_by_typename)
		.def("create_object_copy",&py_kernel::create_object_copy)
		.def("free_instance",&py_kernel::free_instance)
		.def("loaded_plugins",&py_kernel::loaded_plugins)
		.def("plugin_types",&py_kernel::plugin_types)
		.def("plugin_types",&py_kernel::plugin_types_by_name)
		.def("find_type",&py_kernel::find_type)
		.def("register_type",&py_kernel::register_type)
		.def("get_last_error",&py_kernel::get_last_error)
		.def("add_task",&py_kernel::add_task)
		.def("create_storage",&py_kernel::create_storage)
		.def("close_storage",&py_kernel::close_storage)
		.def("load_plugin",&py_kernel::load_plugin)
		.def("unload_plugin",&py_kernel::unload_plugin)
		.def("registered_types",&py_kernel::registered_types)
		.def("bs_root",&py_kernel::bs_root)
		;

	//class_< kernel::plugins_enum >("vector_plugins_enum")
	//	.def(vector_indexing_suite< kernel::plugins_enum >());

	//class_< kernel::types_enum >("vector_types_enum")
	//	.def(vector_indexing_suite< kernel::types_enum >());

	//class_< type_tuple >("type_tuple", init <const plugin_descriptor&, const type_descriptor&>())
	//	.add_property("pd",&type_tuple::pd_)
	//	.add_property("td",&type_tuple::td_)
	//	.def(self == self)
	//	.def(self != self);

	//class_< std::vector< type_tuple > >("vector_type_tuple")
	//	.def(vector_indexing_suite< std::vector< type_tuple > >());
}

}	//namespace blue_sky::python
}	//namespace blue_sky

