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

#ifndef _PY_BS_SHELL_H
#define _PY_BS_SHELL_H

#include "bs_common.h"
#include "bs_shell.h"
#include "py_bs_link.h"

#include <list>

namespace blue_sky {
namespace python {

class BS_API py_deep_iterator {
public:
	typedef std::list< py_bs_link > py_path_t;

	py_deep_iterator();
	py_deep_iterator(const py_deep_iterator &);
	py_deep_iterator(const deep_iterator& i);
	//py_deep_iterator(const smart_ptr< bs_shell, true >& shell);
	//	py_deep_iterator(const std::string &abs_path);
	//py_deep_iterator(const smart_ptr< bs_shell, true >& shell, const std::string& rel_path);

	std::string full_name() const;
	//const py_path_t& path() const;
	//std::string path_name() const;

	py_deep_iterator next();

	bool jump_up();
	bool is_end() const;

private:
	deep_iterator iter;
	py_path_t path_;
};

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _PY_BS_SHELL_H
