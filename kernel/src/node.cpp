/// @file
/// @author uentity
/// @date 15.09.2016
/// @brief BlueSKy tree node implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/node.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

using links_container = node::links_container;
using iterator = node::iterator;
using name_iterator = node::name_iterator;
using name_range = node::name_range;

class node::node_impl {
public:

	sp_link deep_search(const id_type& id) const {
		for(auto& plink : links_) {
			if(plink && plink->id() == id)
				return plink;
		}
		return nullptr;
	}

	links_container links_;
};

node::node()
	: pimpl_(new node_impl())
{}

std::size_t node::size() const {
	return pimpl_->links_.size();
}

bool node::empty() const {
	return pimpl_->links_.empty();
}

iterator node::begin() const {
	return pimpl_->links_.begin();
}

iterator node::end() const {
	return pimpl_->links_.end();
}

name_iterator node::begin_name() const {
	return pimpl_->links_.get< name_key >().begin();
}

name_iterator node::end_name() const {
	return pimpl_->links_.get< name_key >().end();
}

iterator node::find(const id_type& id) const {
	return pimpl_->links_.find(id);
}

iterator node::find(const std::string& name) const {
	return pimpl_->links_.project< 0 >(
		pimpl_->links_.get< name_key >().find(name)
	);
}

iterator node::find(const sp_obj& obj) const {
	iterator elem = begin();
	for(iterator links_end = end(); elem != links_end; ++elem) {
		if(*elem && (*elem)->data() == obj)
			break;
	}
	return elem;
}

name_range node::equal_range(const std::string& name) const {
	return pimpl_->links_.get< name_key >().equal_range(name);
}

name_range node::equal_range(const sp_link& l) const {
	return pimpl_->links_.get< name_key >().equal_range(l->name());
}

node::insert_ret_t node::insert(const sp_link& l) {
	return pimpl_->links_.insert(l);
}

node::insert_ret_t node::insert(const std::string& name, const sp_obj& obj) {
	return pimpl_->links_.insert(
		std::make_shared< hard_link >(name, obj)
	);
}

void node::erase(const std::string& name) {
	pimpl_->links_.get< name_key >().erase(name);
}

void node::erase(const sp_link& l) {
	pimpl_->links_.get< id_key >().erase(l->id());
}

void node::erase(const sp_obj& obj) {
	pimpl_->links_.get< id_key >().erase(find(obj));
}

void node::erase(iterator pos) {
	pimpl_->links_.get< id_key >().erase(pos);
}

void node::erase(iterator from, iterator to) {
	pimpl_->links_.get< id_key >().erase(from, to);
}

sp_link node::deep_search(const id_type& id) const {
	return pimpl_->deep_search(id);
}

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

