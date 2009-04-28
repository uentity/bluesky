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

#ifndef _PY_BS_KERNEL_H
#define _PY_BS_KERNEL_H

#include "bs_kernel.h"
#include "py_bs_object_base.h"
#include "py_bs_command.h"
#include "py_bs_abstract_storage.h"

namespace blue_sky {
namespace python {

class BS_API py_kernel {
private:
	blue_sky::kernel &kernel_;

public:
	py_kernel();
	~py_kernel();

	blue_sky::kernel::plugins_enum loaded_plugins();
	blue_sky::kernel::types_enum plugin_types(const blue_sky::plugin_descriptor& pd);
	blue_sky::kernel::types_enum plugin_types_by_name(const char *pname);

	type_tuple find_type(const char *type_str) const;
	std::vector< type_tuple > registered_types();
	bool register_type(const plugin_descriptor& pd, const type_descriptor& td) const;

	py_objbase create_object(const type_descriptor&);
	py_objbase create_object_by_typename(const char*);

	py_objbase create_object_copy(const py_objbase& src, bool unmanaged = false) const;
	void free_instance(const py_objbase&) const;

	py_bs_abstract_storage create_storage(const char *filename, const char *format, int flags) const;
	void close_storage(const py_bs_abstract_storage &storage) const;

	error_code load_plugin(const std::string& fname, const std::string version);
	void unload_plugin(const plugin_descriptor& pd);

	std::string get_last_error() const;

	void add_task(const py_combase& task);
	py_bs_link bs_root() const;

	void init_logs();
};

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _PY_BS_KERNEL_H
