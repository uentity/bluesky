/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "py_bs_shell.h"
#include "py_bs_exports.h"

using namespace std;

namespace blue_sky {
namespace python {

py_deep_iterator::py_deep_iterator() {}

py_deep_iterator::py_deep_iterator(const py_deep_iterator &titer)
	: iter(titer.iter)
{
	/*for (list< sp_link >::const_iterator i = iter.path().begin(); i != iter.path().end(); ++i)
		path_.push_back(py_bs_link(*i));*/
}

py_deep_iterator::py_deep_iterator(const deep_iterator &titer)
	: iter(titer)
{
	/*for (list< sp_link >::const_iterator i = iter.path().begin(); i != iter.path().end(); ++i)
		path_.push_back(py_bs_link(*i));*/
}
//py_deep_iterator(const smart_ptr< bs_shell, true >& shell);
/*py_deep_iterator::py_deep_iterator(const std::string &abs_path)
	: iter(abs_path)
{
	for (list< sp_link >::const_iterator i = iter.path().begin(); i != iter.path().end(); ++i)
		path_.push_back(py_bs_link(*i));
}*/
//py_deep_iterator(const smart_ptr< bs_shell, true >& shell, const std::string& rel_path);

std::string py_deep_iterator::full_name() const {
	return iter.full_name();
}

/*const py_deep_iterator::py_path_t& py_deep_iterator::path() const {
	return path_;
}*/

/*std::string py_deep_iterator::path_name() const {
	return iter.path_name();
}*/

py_deep_iterator py_deep_iterator::next() {
	++iter;
	return py_deep_iterator(iter);
}

bool py_deep_iterator::jump_up() {
	return iter.jump_up();
}

bool py_deep_iterator::is_end() const {
	return iter.is_end();
}

void py_export_shell() {
	class_<py_deep_iterator>("deep_iter")
		.def("full_name",&py_deep_iterator::full_name)
		.def("next",&py_deep_iterator::next)
		.def("jump_up",&py_deep_iterator::jump_up)
		.def("is_end",&py_deep_iterator::is_end);
}

}	//namespace blue_sky::python
}	//namespace blue_sky
