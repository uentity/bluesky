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

#ifndef _PY_BS_LINK_H
#define _PY_BS_LINK_H

#include "bs_link.h"

namespace blue_sky {
namespace python {

class py_bs_node;
class py_objbase;

class BS_API py_bs_inode {
	friend class py_objbase;

public:
	 typedef smart_ptr< blue_sky::bs_inode, true > sp_inode;
	 //py_bs_inode();
	 py_bs_inode(const sp_inode&);

	 py_objbase *data() const;

	 //info getters
	 ulong size() const;
	 uint uid() const;
	 uint gid() const;
	 uint mode() const;
	 time_t mtime() const;

protected:
	 sp_inode spinode;
};

class BS_API py_bs_link {
	friend class py_bs_node;
public:
	//py_bs_link();
	 py_bs_link(const sp_link&);
	 py_bs_link(const py_bs_link&);
	 //py_bs_link(bs_node::index_type);
	 //py_bs_link(const py_objbase& obj, const py_bs_link& root, bool is_persistent = false);

	 py_bs_inode inode() const;
	 py_objbase data() const;

	 std::string name() const;
	 std::string full_name() const;

	 bool is_node() const;
	 py_bs_node *node() const;

	 //bool is_soft() const;
	 //bool is_persistent() const;
	 /*
	 py_bs_link copy(const py_bs_link& where);
	 bool move(const py_bs_link& where);

	 py_bs_link soft_clone() const;*/

	 bool is_hard_link() const;
	 bs_link::link_type link_type_id() const;

	 py_bs_link clone(const std::string& clone_name = "") const;

	 static py_bs_link dumb_link(const std::string name);

	 //bool subscribe(int signal_code, const sp_slot& slot) const;
	 //bool unsubscribe(int signal_code, const sp_slot& slot) const;

protected:
	 sp_link splink;
};

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _BS_PY_LINK_H
