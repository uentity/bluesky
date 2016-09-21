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

using links_container = bs_node::links_container;
using iterator = bs_node::iterator;
using name_iterator = bs_node::name_iterator;
using name_range = bs_node::name_range;

class bs_node::node_impl {
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

bs_node::bs_node()
	: pimpl_(new node_impl())
{}

std::size_t bs_node::size() const {
	return pimpl_->links_.size();
}

bool bs_node::empty() const {
	return pimpl_->links_.empty();
}

iterator bs_node::begin() const {
	return pimpl_->links_.begin();
}

iterator bs_node::end() const {
	return pimpl_->links_.end();
}

name_iterator bs_node::begin_name() const {
	return pimpl_->links_.get< name_key >().begin();
}

name_iterator bs_node::end_name() const {
	return pimpl_->links_.get< name_key >().end();
}

iterator bs_node::find(const id_type& id) const {
	return pimpl_->links_.find(id);
}

iterator bs_node::find(const std::string& name) const {
	return pimpl_->links_.project< 0 >(
		pimpl_->links_.get< name_key >().find(name)
	);
}

iterator bs_node::find(const sp_obj& obj) const {
	iterator elem = begin();
	for(iterator links_end = end(); elem != links_end; ++elem) {
		if(*elem && (*elem)->data() == obj)
			break;
	}
	return elem;
}

name_range bs_node::equal_range(const std::string& name) const {
	return pimpl_->links_.get< name_key >().equal_range(name);
}

name_range bs_node::equal_range(const sp_link& l) const {
	return pimpl_->links_.get< name_key >().equal_range(l->name());
}

bs_node::insert_ret_t bs_node::insert(const sp_link& l) {
	return pimpl_->links_.insert(l);
}

bs_node::insert_ret_t bs_node::insert(const std::string& name, const sp_obj& obj) {
	return pimpl_->links_.insert(
		std::make_shared< bs_hard_link >(name, obj)
	);
}

void bs_node::erase(const std::string& name) {
	pimpl_->links_.get< name_key >().erase(name);
}

void bs_node::erase(const sp_link& l) {
	pimpl_->links_.get< id_key >().erase(l->id());
}

void bs_node::erase(const sp_obj& obj) {
	pimpl_->links_.get< id_key >().erase(find(obj));
}

void bs_node::erase(iterator pos) {
	pimpl_->links_.get< id_key >().erase(pos);
}

void bs_node::erase(iterator from, iterator to) {
	pimpl_->links_.get< id_key >().erase(from, to);
}

sp_link bs_node::deep_search(const id_type& id) const {
	return pimpl_->deep_search(id);
}

NAMESPACE_END(blue_sky)

