/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
