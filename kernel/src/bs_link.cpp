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

#include "bs_link.h"
#include "bs_tree.h"
#include "bs_exception.h"
#include "bs_object_base.h"
#include "bs_kernel.h"
#include "bs_prop_base.h"

using namespace std;

namespace blue_sky {
//namespace {
//} //end of hidden namespace

////------------------------- unlock listener ---------------------------------
//class bs_inode::obj_listener : public bs_slot {
//public:
//	obj_listener(const bs_inode& i) : i_(i) {}
//
//	void execute(const sp_mobj&, int, const sp_obj&) const {
//		//forward unlock signal to node
//		//DEBUG
//		//cout << "obj_listener at " << this << ": link " << l_.name() << " data_changed signal to be fired" << endl;
//
//		//lock inode first
//		bs_mutex::scoped_lock lk(i_.mutex());
//		//broadcast unlock message to all related hard links
//		for(bs_inode::l_list::const_iterator pl = i_.hl_begin(); pl != i_.hl_end(); ++pl)
//			(*pl)->fire_signal(bs_link::data_changed, NULL);
//	}
//
//	const bs_inode& i_;
//};

//======================== inode implementation ================================================
bs_inode::bs_inode(const sp_obj& obj)
: obj_(obj)
{
	assert(obj_);
	time(&mtime_);
	//ol_ = new obj_listener(*this);
}

sp_obj bs_inode::data() const {
	return obj_;
}

ulong bs_inode::size() const {
	return obj_size_;
}

uint bs_inode::uid() const {
	return uid_;
}

uint bs_inode::gid() const {
	return gid_;
}

uint bs_inode::mode() const {
	return mode_;
}

time_t bs_inode::mtime() const {
	return mtime_;
}

//void bs_inode::del_ref() const {
//	bs_refcounter::del_ref();
//	if(refs() == 1) {
//		//only one reference to this inode in object-container is alive
//		//it means that all hard links to inode is deleted and object is dangling
//		//release it from kernel
//		obj_->bs_free_this();
//	}
//}

void bs_inode::dispose() const {
	delete this;
}

bs_inode::l_list::const_iterator bs_inode::links_begin() const {
	return links_.begin();
}

bs_inode::l_list::const_iterator bs_inode::links_end() const {
	return links_.end();
}

ulong bs_inode::links_count() const {
	return static_cast< ulong >(links_.size());
}

void bs_inode::connect_link(const sp_link& l) {
	links_.insert(l);
}

void bs_inode::disconnect_link(const sp_link& l) {
	links_.erase(l);
	if(links_.empty()) {
		//all hard links to inode is deleted and object is dangling
		//delete it from kernel
		obj_->bs_free_this();
	}
}

//========================== link_impl abstract definition ======================================
namespace {
//------------------------- unlock listener ---------------------------------
class obj_listener : public bs_slot {
public:
	obj_listener(const bs_link* l) : l_(l) {}

	void execute(const sp_mobj&, int, const sp_obj&) const {
		//forward unlock signal to node
		//DEBUG
		//cout << "obj_listener at " << this << ": link " << l_.name() << " data_changed signal to be fired" << endl;

		l_->fire_signal(bs_link::data_changed, NULL);
	}

	const bs_link* l_;
};

}	//end of hidden namspace

class bs_link::link_impl {
public:
	std::string name_;
	//bool is_persistent_;
	const bs_link* self_;
	//unlock signal forwarder
	sp_slot ol_;
	bool is_listening_;


	link_impl(const std::string& name)
		: name_(name), is_listening_(false)
	{}

	virtual sp_inode inode() const = 0;
	//virtual void set_inode(const sp_inode& i) = 0;

	virtual const std::string& name() const {
		return name_;
	}

	virtual void rename(const std::string& new_name) {
		name_ = new_name;
	}

	//virtual bool is_persistent() const {
	//	return is_persistent_;
	//}

	virtual void listen2obj() {
		if(!is_listening_ && inode() && self_) {
			if(!ol_) ol_ = new obj_listener(self_);
			inode()->data()->subscribe(objbase::on_unlock, ol_);
			is_listening_ = true;
		}
	}

	virtual void stop_listening() {
		//stop listening for object unlock signal
		if(is_listening_ && inode() && ol_) {
			inode()->data()->unsubscribe(objbase::on_unlock, ol_);
			is_listening_ = false;
		}
	}

	virtual void set_owner(const bs_link* l, bool start_listening = false) {
		//bool listen_status = is_listening_;
		stop_listening();
		//set owner
		self_ = l;
		//recreate object listner connected to new link
		if(start_listening)
			listen2obj();
	}

	//virtual void swap(link_impl* impl) = 0;

	//virtual dtor
	virtual ~link_impl() {
		stop_listening();
	};

private:
	//copy construction is denied
	link_impl(const link_impl&);
};

//========================== hard link implementation ===============================================
//----------------------- hard link implementation --------------------------------------------
class bs_link::hl_impl : public link_impl
{
private:
	//main task of this class is to connect to inode on construction and disconnect on desctruction
	struct inode_tracker : public bs_refcounter {
		inode_tracker(const bs_link* pl) : plink_(pl) {
			//impl->self_ = pl;
			if(pl->inode())
				pl->inode().lock()->connect_link(pl);
		}

		~inode_tracker() {
			if(plink_->inode())
				plink_->inode().lock()->disconnect_link(plink_);
		}

		void dispose() const {
			delete this;
		}

		const bs_link* plink_;
	};

	smart_ptr< inode_tracker > it_;

public:
	//member variables
	sp_inode inode_;
//	sp_slot ol_;
//	bool is_listening_;

	hl_impl(const string& name, const sp_obj& obj = NULL)
		: link_impl(name) //, is_listening_(false)
	{
		if(obj) {
			//if object doesn't have an inode then create one
			if(!obj->inode_)
				obj.lock()->inode_ = new blue_sky::bs_inode(obj);

			set_inode(obj->inode());
		}
	}

	hl_impl(const hl_impl& impl)
		: link_impl(impl.name()) //, is_listening_(false)
	{
		set_inode(impl.inode());
	}

	void set_owner(const bs_link* l, bool start_listening = false) {
		// call parent
		link_impl::set_owner(l, start_listening);
		// track inode's hard link's list
		it_ = new inode_tracker(l);
	}

	sp_inode inode() const {
		return inode_;
	}

	void set_inode(const sp_inode& i) {
		//bool listen_status = is_listening_;
		stop_listening();
		//switch inode
		if(!i) {
			//reset pimpl
			inode_ = NULL;
		//	return;
		}
		else {
			inode_ = i;
			//subscribe to object's unlock event
			//connect_node_handler(node_hr_);
		}
		//if we were listening - start listening for new object
		//if(listen_status && ol_)
		//	listen2obj(((obj_listener*)ol_.get())->l_);
	}

	~hl_impl() {}

	/*	sp_node root() const {
	return root_->node();
	}*/
};

//============================ alias implementation ====================================================================
class bs_alias::sl_impl : public link_impl {
public:
	sp_link link_;
//	sp_slot ol_;
//	bool is_listening_;

	sl_impl(const sp_link& link, const string& name)
		: link_impl(name), link_(link)
	{
		set_link(link);
	}

	sp_inode inode() const {
		if(link_)
			return link_->inode();
		else return NULL;
	}

	void set_link(const sp_link& l) {
		//bool listen_status = is_listening_;
		stop_listening();
		//switch inode
		if(!l) {
			//reset pimpl
			link_ = NULL;
			//return;
		}
		else {
			link_ = l;
		}
		//if we were listening - start listening for new object
		//if(listen_status && ol_)
		//	listen2obj(((obj_listener*)ol_.get())->l_);
	}

	~sl_impl() {};
};

//================================== bs_alias implementation ===========================================================
bs_alias::bs_alias(bs_type_ctor_param /*p*/)
	//: bs_link(new sl_impl(NULL, "", false))
{
	throw bs_exception("BlueSky kernel", "Use bs_link::create to create links");
}

//standard ctor
bs_alias::bs_alias(const sp_link& link, const std::string& name)
	: bs_link(new sl_impl(link, name))
{
	assert(pimpl_->inode());
	pimpl_.lock()->set_owner(this);
}

objbase* bs_alias::bs_create_instance(blue_sky::bs_type_ctor_param param) {
	smart_ptr< str_data_table > sp_dt(param, bs_dynamic_cast());
	if(!sp_dt) throw bs_exception("BlueSky kernel", "No parameters were passed to bs_alias constructor");
	sp_link link = sp_dt->extract_value< sp_link >("link");
	string name = sp_dt->extract_value< string >("name");
	return new bs_alias(link, name);
}

sp_link bs_alias::create(const sp_link& link, const std::string& name) {
	lsmart_ptr< smart_ptr< str_data_table > > sp_dt(BS_KERNEL.create_object(str_data_table::bs_type(), true));
	if(!sp_dt)
		throw bs_exception("BlueSky kernel", "Unable to create alias - str_val_table creation failed");
	sp_dt->add_item< sp_obj >("link", link);
	sp_dt->add_item< string >("name", name);
	return BS_KERNEL.create_object(bs_type(), false, sp_dt);
}

string bs_alias::name() const {
	return pimpl_->name();
}

sp_link bs_alias::clone(const string& clone_name) const {
	//return new bs_alias(((sl_impl*)pimpl_.get())->link_, name(), is_persistent);
	if(clone_name == "")
		return new bs_alias(((sl_impl*)pimpl_.get())->link_, name());
	else
		return new bs_alias(((sl_impl*)pimpl_.get())->link_, clone_name);
}

bs_link::link_type bs_alias::link_type_id() const {
	return alias;
}

//================================== bs_link implementation ============================================================
bs_link::bs_link(bs_type_ctor_param param)
	: objbase(BS_SIGNAL_RANGE(bs_link)),
	  pimpl_((hl_impl*)NULL, mutex(), bs_static_cast())
{
	smart_ptr< str_data_table > sp_dt(param, bs_dynamic_cast());
	if(sp_dt) {
		sp_obj obj = sp_dt->extract_value< sp_obj >("object");
		if(!obj)
			throw bs_exception("BlueSky kernel", "NULL object pointer passed to blue_sky::link constructor");
		string name = sp_dt->extract_value< std::string >("name");
		//if object doesn't have an inode then create one
		if(!obj->inode_)
			obj.lock()->inode_ = new blue_sky::bs_inode(obj);
		//create default implementation
		pimpl_ = new hl_impl(name, obj);
		//connect this link to corresponding inode
		pimpl_.lock()->set_owner(this);
	}
	else
		throw bs_exception("BlueSky kernel", "No parameters were passed to blue_sky::link constructor");
}

sp_link bs_link::create(const sp_obj& obj, const std::string& name) {
	lsmart_ptr< smart_ptr< str_data_table > > sp_dt(BS_KERNEL.create_object(str_data_table::bs_type(), true));
	if(!sp_dt)
		throw bs_exception("BlueSky kernel", "Unable to create link - str_val_table creation failed");
	sp_dt->add_item< sp_obj >("object", obj);
	sp_dt->add_item< string >("name", name);
	return BS_KERNEL.create_object(bs_type(), false, sp_dt);
}

bs_link::bs_link(const sp_obj& obj, const std::string& name)
	: objbase(BS_SIGNAL_RANGE(bs_link)),
	  pimpl_((hl_impl*)NULL, mutex(), bs_static_cast())
	  //pimpl_(new hl_impl(name, is_persistent, obj), mutex(), bs_static_cast())
{
	if(!obj) throw bs_exception("BlueSky kernel", "NULL object pointer passed to blue_sky::link constructor");
	//if object doesn't have an inode then create one
	if(!obj->inode_)
		obj.lock()->inode_ = new blue_sky::bs_inode(obj);
	//create default implementation
	pimpl_ = new hl_impl(name, obj);

	//connect this link to corresponding inode
	pimpl_.lock()->set_owner(this);
}

//ctor for childs
bs_link::bs_link(const link_impl* impl)
	: objbase(BS_SIGNAL_RANGE(bs_link)),
	  pimpl_(impl, mutex(), bs_static_cast())
{
	pimpl_.lock()->set_owner(this);
}

//protected copy ctor
//bs_link::bs_link(const blue_sky::bs_link &l)
//: pimpl_(new hl_impl(*l.pimpl_), mutex(), bs_static_cast())
//{}

//void bs_link::swap(bs_link& l) const {
//	pimpl_.lock()->swap(l.pimpl_);
//}

//bs_link& bs_link::operator=(const bs_link& l) {
//	//assignment through swap
//	bs_link(l).swap(*this);
//	return *this;
//}

//destructor
bs_link::~bs_link() {
	//cout << "bs_link '" << name() << "' dtor entered" << endl;
}

//deletition method
void bs_link::dispose() const {
	delete this;
}

sp_link bs_link::clone(const string& clone_name) const {
	if(clone_name == "")
		return new bs_link(data(), name());
	else
		return new bs_link(data(), clone_name);
}

sp_inode bs_link::inode() const {
	return pimpl_->inode();
}

sp_obj bs_link::data() const {
	if(pimpl_->inode())
		return pimpl_->inode()->data();
	else return NULL;
}

std::string bs_link::name() const {
	return pimpl_->name();
}

// std::string bs_link::full_name() const {
// 	return pimpl_->root()->bs_name() + "/" + name();
// }

bool bs_link::is_node() const {
	return inode() && bs_node::is_node(pimpl_->inode()->data());
}

sp_node bs_link::node() const {
	if(inode())
		return sp_node(pimpl_->inode()->data(), bs_dynamic_cast());
	return NULL;
}


sp_link bs_link::dumb_link(const std::string name) {
	return new bs_link(new hl_impl(name, NULL));
}

void bs_link::rename(const std::string& new_name) const {
	pimpl_.lock()->rename(new_name);
}

bool bs_link::is_hard_link() const {
	return (link_type_id() == hard_link);
	//return (BS_GET_TI(*pimpl_) == BS_GET_TI(hl_impl));
}

bs_link::link_type bs_link::link_type_id() const {
	return hard_link;
}

bool bs_link::subscribe(int signal_code, const sp_slot& slot) const {
	bool res = bs_messaging::subscribe(signal_code, slot);
	// listen to object only when at least anybody subscribed
	if(res && signal_code == data_changed)
		pimpl_.lock()->listen2obj();
	return res;
}

//signals manipulation for bs_node
bool bs_link::unsubscribe(int signal_code, const sp_slot& handler) const {
	bool res = bs_messaging::unsubscribe(signal_code, handler);
	//stop listening for underlying object if everybody has unsubsribed
	if(signal_code == data_changed && num_slots(signal_code) == 0) {
		pimpl_.lock()->stop_listening();
	}
	return res;
}

BLUE_SKY_TYPE_STD_CREATE(bs_link)
BLUE_SKY_TYPE_IMPL_NOCOPY_SHORT(bs_link, objbase, "Represent link in object's tree")

BLUE_SKY_TYPE_IMPL_NOCOPY_SHORT(bs_alias, bs_link, "Represent a direct reference to given link in object's tree")

} //end of namespace blue_sky
