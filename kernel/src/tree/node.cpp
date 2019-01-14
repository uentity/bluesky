/// @file
/// @author uentity
/// @date 15.09.2016
/// @brief BlueSKy tree node implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_impl.h"
#include <bs/tree/tree.h>
#include <bs/kernel/types_factory.h>

#include <boost/uuid/string_generator.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

NAMESPACE_BEGIN() // hidden namespace

static boost::uuids::string_generator uuid_from_str;

NAMESPACE_END()

using links_locker_t = std::lock_guard<std::mutex>;
/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
node::node(std::string custom_id)
	: objbase(true, custom_id), pimpl_(std::make_unique<node_impl>())
{}

node::node(const node& src)
	: objbase(src), pimpl_(std::make_unique<node_impl>(*src.pimpl_))
{}

node::~node() = default;

void node::propagate_owner(bool deep) {
	links_locker_t my_turn(pimpl_->links_guard_);
	// properly setup owner in node's leafs
	const auto self = bs_shared_this<node>();
	sp_node child_node;
	for(auto& plink : pimpl_->links_) {
		child_node = node_impl::adjust_inserted_link(plink, self);
		if(deep && child_node)
			child_node->propagate_owner(true);
	}
}

sp_link node::handle() const {
	return pimpl_->handle_.lock();
}

//sp_link node::create_self_link(std::string name, bool force) {
//	if(!force && !pimpl_->handle_.expired()) {
//		return pimpl_->handle_.lock();
//	}
//	// create new self link
//	auto root_lnk = std::make_shared<hard_link>(std::move(name), bs_shared_this<node>());
//	pimpl_->set_handle(root_lnk);
//	return root_lnk;
//}


std::size_t node::size() const {
	return pimpl_->links_.size();
}

bool node::empty() const {
	return pimpl_->links_.empty();
}

// ---- begin/end
iterator<Key::AnyOrder> node::begin(Key_const<Key::AnyOrder>) const {
	return pimpl_->begin<>();
}

iterator<Key::AnyOrder> node::end(Key_const<Key::AnyOrder>) const {
	return pimpl_->end<>();
}

iterator<Key::ID> node::begin(Key_const<Key::ID>) const {
	return pimpl_->begin<Key::ID>();
}

iterator<Key::ID> node::end(Key_const<Key::ID>) const {
	return pimpl_->end<Key::ID>();
}

iterator<Key::Name> node::begin(Key_const<Key::Name>) const {
	return pimpl_->begin<Key::Name>();
}

iterator<Key::Name> node::end(Key_const<Key::Name>) const {
	return pimpl_->end<Key::Name>();
}

iterator<Key::OID> node::begin(Key_const<Key::OID>) const {
	return pimpl_->begin<Key::OID>();
}

iterator<Key::OID> node::end(Key_const<Key::OID>) const {
	return pimpl_->end<Key::OID>();
}

iterator<Key::Type> node::begin(Key_const<Key::Type>) const {
	return pimpl_->begin<Key::Type>();
}

iterator<Key::Type> node::end(Key_const<Key::Type>) const {
	return pimpl_->end<Key::Type>();
}

// ---- find
iterator<Key::AnyOrder> node::find(const std::size_t idx) const {
	auto i = begin();
	std::advance(i, idx);
	return i;
}

iterator<Key::AnyOrder> node::find(const id_type& id) const {
	return pimpl_->find<Key::ID>(id);
}

iterator<Key::AnyOrder> node::find(const std::string& key, Key key_meaning) const {
	switch(key_meaning) {
	case Key::ID:
		return pimpl_->find<Key::ID>(uuid_from_str(key));
	case Key::OID:
		return pimpl_->find<Key::OID>(key);
	case Key::Type:
		return pimpl_->find<Key::Type>(key);
	default:
	case Key::Name:
		return pimpl_->find<Key::Name>(key);
	}
}

// ---- index
std::size_t node::index(const id_type& lid) const {
	auto i = pimpl_->find<Key::ID, Key::AnyOrder>(lid);
	return std::distance(begin<Key::AnyOrder>(), i);
}

std::size_t node::index(const iterator<Key::AnyOrder>& pos) const {
	return std::distance(begin<Key::AnyOrder>(), pos);
}

std::size_t node::index(const std::string& key, Key key_meaning) const {
	auto i = find(key, key_meaning);
	return std::distance(begin<Key::AnyOrder>(), i);
}

// ---- equal_range
range<Key::Name> node::equal_range(const std::string& link_name) const {
	return pimpl_->equal_range<Key::Name>(link_name);
}

range<Key::OID> node::equal_range_oid(const std::string& oid) const {
	return pimpl_->equal_range<Key::OID>(oid);
}

range<Key::Type> node::equal_type(const std::string& type_id) const {
	return pimpl_->equal_range<Key::Type>(type_id);
}

// ---- insert
insert_status<Key::ID> node::insert(sp_link l, InsertPolicy pol) {
	auto res = pimpl_->insert(l, pol);
	if(res.second) {
		// inserted link postprocessing
		node_impl::adjust_inserted_link(*res.first, bs_shared_this<node>());
	}
	else if(enumval(pol & InsertPolicy::Merge) && res.first != end<Key::ID>()) {
		// check if we need to deep merge given links
		// go one step down the hierarchy
		auto src_node = l->data_node();
		auto dst_node = (*res.first)->data_node();
		if(src_node && dst_node) {
			// insert all links from source node into destination
			dst_node->insert(
				std::vector<sp_link>(src_node->begin(), src_node->end()), pol
			);
		}
	}
	return res;
}

insert_status<Key::AnyOrder> node::insert(sp_link l, iterator<> pos, InsertPolicy pol) {
	// 1. insert an element using ID index
	auto res = insert(std::move(l), pol);
	if(res.first != end<Key::ID>()) {
		// 2. reposition an element in AnyOrder index
		links_locker_t my_turn(pimpl_->links_guard_);
		auto src = pimpl_->project<Key::ID>(res.first);
		if(pos != src) {
			auto& ord_idx = pimpl_->links_.get<Key_tag<Key::AnyOrder>>();
			ord_idx.relocate(pos, src);
			return {src, true};
		}
	}
	return {pimpl_->project<Key::ID>(res.first), res.second};
}

insert_status<Key::AnyOrder> node::insert(sp_link l, std::size_t idx, InsertPolicy pol) {
	return insert(std::move(l), std::next(begin<Key::AnyOrder>(), std::min(idx, size())), pol);
}

insert_status<Key::ID> node::insert(std::string name, sp_obj obj, InsertPolicy pol) {
	return insert(
		std::make_shared<hard_link>(std::move(name), std::move(obj)), pol
	);
}

// ---- erase
void node::erase(const std::size_t idx) {
	links_locker_t my_turn(pimpl_->links_guard_);
	pimpl_->links_.get<Key_tag<Key::AnyOrder>>().erase(find(idx));
}

void node::erase(const id_type& lid) {
	pimpl_->erase<>(lid);
}

void node::erase(const std::string& key, Key key_meaning) {
	switch(key_meaning) {
	case Key::ID:
		pimpl_->erase<Key::ID>(uuid_from_str(key));
		break;
	case Key::OID:
		pimpl_->erase<Key::OID>(key);
		break;
	case Key::Type:
		pimpl_->erase<Key::Type>(key);
		break;
	default:
	case Key::Name:
		return pimpl_->erase<Key::Name>(key);
	}
}

// ---- erase range
void node::erase(const range<Key::ID>& r) {
	pimpl_->erase<>(r);
}

void node::erase(const range<Key::Name>& r) {
	pimpl_->erase<Key::Name>(r);
}

void node::erase(const range<Key::OID>& r) {
	pimpl_->erase<Key::OID>(r);
}

void node::clear() {
	links_locker_t my_turn(pimpl_->links_guard_);
	pimpl_->links_.clear();
}

// ---- deep_search
sp_link node::deep_search(const id_type& id) const {
	return pimpl_->deep_search<>(id);
}

sp_link node::deep_search(const std::string& key, Key key_meaning) const {
	switch(key_meaning) {
	case Key::ID:
		return pimpl_->deep_search<Key::ID>(uuid_from_str(key));
	case Key::OID:
		return pimpl_->deep_search<Key::OID>(key);
	case Key::Type:
		return pimpl_->deep_search<Key::Type>(key);
	default:
	case Key::Name:
		return pimpl_->deep_search<Key::Name>(key);
	}
}


// ---- keys
std::vector<Key_type<Key::ID>> node::keys(Key_const<Key::ID>) const {
	return pimpl_->keys<Key::ID>();
}

std::vector<Key_type<Key::Name>> node::keys(Key_const<Key::Name>) const {
	return pimpl_->keys<Key::Name>();
}

std::vector<Key_type<Key::OID>> node::keys(Key_const<Key::OID>) const {
	return pimpl_->keys<Key::OID>();
}

std::vector<Key_type<Key::Type>> node::keys(Key_const<Key::Type>) const {
	return pimpl_->keys<Key::Type>();
}

// ---- rename
bool node::rename(iterator<Key::AnyOrder> pos, std::string new_name) {
	return pimpl_->rename<Key::AnyOrder>(std::move(pos), std::move(new_name));
}

bool node::rename(const id_type& lid, std::string new_name) {
	return pimpl_->rename<Key::ID>(lid, std::move(new_name)) > 0;
}

int node::rename(const std::string& key, std::string new_name, Key key_meaning, bool all) {
	switch(key_meaning) {
	default:
	case Key::ID:
		return pimpl_->rename<Key::ID>(uuid_from_str(key), std::move(new_name), all);
	case Key::OID:
		return pimpl_->rename<Key::OID>(key, std::move(new_name), all);
	case Key::Type:
		return pimpl_->rename<Key::Type>(key, std::move(new_name), all);
	case Key::Name:
		return pimpl_->rename<Key::Name>(key, std::move(new_name), all);
	}
}

auto node::on_rename(const id_type& renamed_id) const -> void {
	pimpl_->on_rename(renamed_id);
}

// ---- project
iterator<Key::AnyOrder> node::project(iterator<Key::ID> src) const {
	return pimpl_->project<Key::ID>(std::move(src));
}

iterator<Key::AnyOrder> node::project(iterator<Key::Name> src) const {
	return pimpl_->project<Key::Name>(std::move(src));
}

iterator<Key::AnyOrder> node::project(iterator<Key::OID> src) const {
	return pimpl_->project<Key::OID>(std::move(src));
}

iterator<Key::AnyOrder> node::project(iterator<Key::Type> src) const {
	return pimpl_->project<Key::Type>(std::move(src));
}

// ---- other
bool node::accepts(const sp_link& what) const {
	return pimpl_->accepts(what);
}

void node::accept_object_types(std::vector<std::string> allowed_types) {
	pimpl_->allowed_otypes_ = std::move(allowed_types);
}

std::vector<std::string> node::allowed_object_types() const {
	return pimpl_->allowed_otypes_;
}

void node::set_handle(const sp_link& handle) {
	pimpl_->set_handle(handle);
}

BS_TYPE_IMPL(node, objbase, "node", "BS tree node", true, true);
BS_TYPE_ADD_CONSTRUCTOR(node, (std::string))
BS_REGISTER_TYPE("kernel", node)

NAMESPACE_END(tree)

namespace detail {

BS_API void adjust_cloned_node(const sp_obj& pnode) {
	// fix owner in node's clone
	if(pnode->is_node())
		std::static_pointer_cast<tree::node>(pnode)->propagate_owner();
}
	
} /* namespace detail */

NAMESPACE_END(blue_sky)

