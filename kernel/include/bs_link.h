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

#ifndef _BS_LINK_H
#define _BS_LINK_H

#include "bs_common.h"
#include "bs_object_base.h"
#include "bs_command.h"
#include <time.h>

namespace blue_sky {

class bs_node;
class bs_link;

class BS_API bs_inode : virtual public bs_refcounter {
	friend class objbase;
	friend class kernel;
	//friend class bs_node;
	friend class bs_link;

public:
	//type of list of hard links to this inode
	typedef std::set< smart_ptr< bs_link, true > > l_list;

	//member functions
	//access to pointed object
	sp_obj data() const;

	//info getters
	ulong size() const;
	uint uid() const;
	uint gid() const;
	uint mode() const;
	time_t mtime() const;

	//override refcounter decrement method in order to track hard links number
	//void del_ref() const;

	//access to hard links list
	l_list::const_iterator links_begin() const;
	l_list::const_iterator links_end() const;
	ulong links_count() const;

private:
	//pointer to corresponding object
	sp_obj obj_;

	//hard links list
	l_list links_;
	//unlock message forwarder
//	class obj_listener;
//	smart_ptr< obj_listener, true > ol_;

	//like POSIX inode
	//size of stored object
	ulong obj_size_;
	//UID of owner
	uint uid_;
	//GID
	uint gid_;
	//object mode - specifies access
	uint mode_;
	//timestamp of last modification time
	time_t mtime_;

	//special ctor for derived classes
	bs_inode(const sp_obj& obj);
	//inode destruction method
	void dispose() const;

	//connect hard link
	void connect_link(const smart_ptr< bs_link, true >& l);
	//disconnect hard link
	void disconnect_link(const smart_ptr< bs_link, true >& l);
};
//global typedef
typedef smart_ptr< bs_inode > sp_inode;

class BS_API bs_link : public objbase { //virtual public bs_refcounter, public bs_messaging {
	friend class kernel;
	friend class bs_node;

public:
	enum link_type {
		hard_link = 1,
		alias,
		sym_link
	};

	//signals definition
	BLUE_SKY_SIGNALS_DECL_BEGIN(objbase)
		data_changed,
		inode_changed,
		//link_renamed,
	BLUE_SKY_SIGNALS_DECL_END

	typedef smart_ptr< bs_link, true > sp_link;

	//links creation method
	static sp_link create(const sp_obj& obj, const std::string& name);

	//virtual dtor
	virtual ~bs_link();

	//inode getter
	sp_inode inode() const;
	//object behind inode accessor - forwards call to bs_inode.data()
	sp_obj data() const;

	//name of link
	virtual std::string name() const;

	//full path to object
	//virtual std::string full_name() const;

	//test if pointed object is an instance of bs_node
	virtual bool is_node() const;
	//if pointed object is a node - returns non-NULL SP to it
	smart_ptr< blue_sky::bs_node, true > node() const;

	//tests if this is an instance of bs_link, i.e. direct hard link to object
	bool is_hard_link() const;
	//return type id of link
	virtual link_type link_type_id() const;

	//clone method for links
	virtual sp_link clone(const std::string& clone_name = "") const;
	//alias creation
	//virtual sp_link alias(const std::string& name = "", bool is_persistent = false) const;

	//generates link that points to nothing only for searching purposes
	static sp_link dumb_link(const std::string name);

	//comparison operator
	//bool operator==(const bs_link& l) const;

	bool subscribe(int signal_code, const sp_slot& slot) const;
	bool unsubscribe(int signal_code, const sp_slot& slot) const;

protected:
	//link's pimpl
	class link_impl;
	class hl_impl;
	smart_ptr< link_impl, false > pimpl_;

	//ctors
	//construct from object, i.e. from inode - makes hard link
	bs_link(const sp_obj& obj, const std::string& name);

	//protected ctor for children
	bs_link(const link_impl* impl);

	//virtual destruction method
	virtual void dispose() const;

	//rebase link so that it starts pointing to another object
	void rebase(const sp_obj& obj) const;

	//swaps 2 links - needed for assignment
	//void swap(bs_link& l) const;
	//assignment - for bs_tree only
	//bs_link& operator=(const bs_link& l);

private:
	//copies creation is prohibited - use clone() instead
	//bs_link(const bs_link& l);

	//renamer for bs_node
	virtual void rename(const std::string& new_name) const;

	BLUE_SKY_TYPE_DECL(bs_link)
};
//global typedef
typedef bs_link::sp_link sp_link;

class BS_API bs_alias : public bs_link {
	friend class bs_link;

public:
	static sp_link create(const sp_link& link, const std::string& name);

	//name accessor
	std::string name() const;

	sp_link clone(const std::string& clone_name = "") const;

	bs_link::link_type link_type_id() const;

private:
	class sl_impl;

	//standard ctor
	bs_alias(const sp_link& link, const std::string& name);

	//void set_persistence(bool state = true) const;

	//bs_alias is noncopyable for clients
	//bs_alias(const bs_alias&);

	BLUE_SKY_TYPE_DECL(bs_alias)
};

} //end of blue_sky namespace

#endif
