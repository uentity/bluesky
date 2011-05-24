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
#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

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
