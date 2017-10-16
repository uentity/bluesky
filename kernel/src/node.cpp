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
using Key = node::Key;
using name_range = node::name_range;
template<Key K> using iterator = typename node::iterator<K>;
template<Key K> using Key_tag = typename node::Key_tag<K>;
template<Key K> using Key_type = typename node::Key_type<K>;
template<Key K> using Key_const = typename node::Key_const<K>;

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

iterator<Key::ID> node::begin(Key_const<Key::ID>) const {
	return pimpl_->links_.begin();
}

iterator<Key::ID> node::end(Key_const<Key::ID>) const {
	return pimpl_->links_.end();
}

iterator<Key::Name> node::begin(Key_const<Key::Name>) const {
	return pimpl_->links_.get< name_key >().begin();
}

iterator<Key::Name> node::end(Key_const<Key::Name>) const {
	return pimpl_->links_.get< name_key >().end();
}

iterator<Key::ID> node::find(const id_type& id) const {
	return pimpl_->links_.find(id);
}

iterator<Key::ID> node::find(const std::string& name) const {
	return pimpl_->links_.project< 0 >(
		pimpl_->links_.get< name_key >().find(name)
	);
}

iterator<Key::ID> node::find(const sp_obj& obj) const {
	using iterator = iterator<Key::ID>;
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

node::insert_ret_t<Key::ID> node::insert(const sp_link& l) {
	return pimpl_->links_.insert(l);
}

node::insert_ret_t<Key::ID> node::insert(const std::string& name, const sp_obj& obj) {
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

void node::erase(iterator<Key::ID> pos) {
	pimpl_->links_.get< id_key >().erase(pos);
}

void node::erase(iterator<Key::ID> from, iterator<Key::ID> to) {
	pimpl_->links_.get< id_key >().erase(from, to);
}

sp_link node::deep_search(const id_type& id) const {
	return pimpl_->deep_search(id);
}

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

