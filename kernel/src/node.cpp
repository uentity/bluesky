/// @file
/// @author uentity
/// @date 15.09.2016
/// @brief BlueSKy tree node implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/node.h>
#include <bs/kernel.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

using links_container = node::links_container;
using Key = node::Key;
template<Key K> using iterator = typename node::iterator<K>;
template<Key K> using Key_tag = typename node::Key_tag<K>;
template<Key K> using Key_type = typename node::Key_type<K>;
template<Key K> using Key_const = typename node::Key_const<K>;
template<Key K> using insert_status = typename node::insert_status<K>;
template<Key K> using range = typename node::range<K>;

/*-----------------------------------------------------------------------------
 *  node_impl
 *-----------------------------------------------------------------------------*/
class node::node_impl {
public:

	template<Key K = Key::ID>
	static sp_link deep_search_impl(const node_impl& n, const Key_type<K>& key) {
		// first do direct search in leafs
		auto r = n.find<K, K>(key);
		if(r != n.end<K>()) return *r;

		// if not succeeded search in children nodes
		for(const auto& plink : n.links_) {
			if(plink->obj_type_id() == node::bs_type().name) {
				auto l = deep_search_impl<K>(
					*std::static_pointer_cast<node>(plink->data())->pimpl_, key
				);
				if(l) return l;
			}
		}
		return nullptr;
	}

	//static deep_merge(const node_impl& n, )

	template<Key K = Key::ID>
	auto begin() const {
		return links_.get<Key_tag<K>>().begin();
	}
	template<Key K = Key::ID>
	auto end() const {
		return links_.get<Key_tag<K>>().end();
	}

	template<Key K, Key R = Key::ID>
	auto find(
		const Key_type<K>& key,
		std::enable_if_t<std::is_same<Key_const<K>, Key_const<Key::ID>>::value>* = nullptr
	) const {
		return links_.get<Key_tag<K>>().find(key);
	}

	template<Key K, Key R = Key::ID>
	auto find(
		const Key_type<K>& key,
		std::enable_if_t<!std::is_same<Key_const<K>, Key_const<Key::ID>>::value>* = nullptr
	) const {
		return links_.project<Key_tag<R>>(
			links_.get<Key_tag<K>>().find(key)
		);
	}

	template<Key K = Key::ID>
	auto equal_range(const Key_type<K>& key) const {
		return links_.get<Key_tag<K>>().equal_range(key);
	}

	template<Key K = Key::ID>
	void erase(const Key_type<K>& key) {
		links_.get<Key_tag<K>>().erase(key);
	}

	template<Key K = Key::ID>
	void erase(const range<K>& r) {
		links_.get<Key_tag<K>>().erase(r.first, r.second);
	}

	template<Key K = Key::ID>
	sp_link deep_search(const Key_type<K>& key) const {
		return this->deep_search_impl<K>(*this, key);
	}

	insert_status<Key::ID> insert(sp_link l, uint pol) {
		// check if we have duplication name
		iterator<Key::ID> dup;
		if((pol & 3) > 0) {
			dup = find<Key::Name>(l->name());
			if(dup != end<>()) {
				// first check if dup names are prohibited
				if(pol & InsertPolicy::DenyDupNames) return {dup, false};
				else {
					// try to auto-rename link
					std::string new_name;
					bool unique_found = false;
					for(int i = 0; i < 1000; ++i) {
						new_name = l->name() + '_' + std::to_string(i);
						if(find<Key::Name, Key::Name>(new_name) == end<Key::Name>()) {
							// we've found a unique name
							l->name_ = std::move(new_name);
							unique_found = true;
							break;
						}
					}
					// if no unique name was found - return fail
					if(!unique_found) return {dup, false};
				}
			}
		}
		// check for duplicating OID
		if(pol & InsertPolicy::DenyDupNames) {
			dup = find<Key::OID>(l->oid());
			if(dup != end<>()) return {dup, false};
		}
		// try to insert given link
		return links_.insert(std::move(l));
	}

	// implement deep copy ctor
	node_impl(const node_impl& src) {
		for(const auto& plink : src.links_) {
			links_.insert(plink->clone());
		}
	}

	node_impl() = default;

	links_container links_;
};

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

std::size_t node::size() const {
	return pimpl_->links_.size();
}

bool node::empty() const {
	return pimpl_->links_.empty();
}

iterator<Key::ID> node::begin(Key_const<Key::ID>) const {
	return pimpl_->begin<>();
}

iterator<Key::ID> node::end(Key_const<Key::ID>) const {
	return pimpl_->end<>();
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

iterator<Key::ID> node::find(const id_type& id) const {
	return pimpl_->find<Key::ID>(id);
}

iterator<Key::ID> node::find(const std::string& name) const {
	return pimpl_->find<Key::Name>(name);
}

iterator<Key::ID> node::find_oid(const std::string& oid) const {
	return pimpl_->find<Key::OID>(oid);
}

range<Key::Name> node::equal_range(const std::string& link_name) const {
	return pimpl_->equal_range<Key::Name>(link_name);
}

range<Key::OID> node::equal_range_oid(const std::string& oid) const {
	return pimpl_->equal_range<Key::OID>(oid);
}

insert_status<Key::ID> node::insert(sp_link l, uint pol) {
	auto res = pimpl_->insert(std::move(l), pol);
	// switch owner to *this if succeeded
	if(res.second) {
		auto& res_lnk = *(res.first->get());
		if(auto prev_owner = res_lnk.owner())
			prev_owner->erase(res_lnk.id());
		res_lnk.reset_owner(bs_shared_this<node>());
	}
	else if(pol | InsertPolicy::Merge && res.first != end()) {
		// check if we need to deep merge given links
		// go one step down the hierarchy
		auto src_node = l->data_node();
		auto dst_node = (*res.first)->data_node();
		if(src_node && dst_node) {
			// insert all links from source node into destination
			dst_node->insert(*src_node, pol);
		}
	}
	return res;
}

insert_status<Key::ID> node::insert(std::string name, sp_obj obj, uint pol) {
	return insert(
		std::make_shared< hard_link >(std::move(name), std::move(obj)), pol
	);
}

void node::erase(const id_type& lid) {
	pimpl_->erase<>(lid);
}

void node::erase(const std::string& name) {
	pimpl_->erase<Key::Name>(name);
}

void node::erase_oid(const std::string& oid) {
	pimpl_->erase<Key::OID>(oid);
}

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
	pimpl_->links_.clear();
}

sp_link node::deep_search(const id_type& id) const {
	return pimpl_->deep_search<>(id);
}

sp_link node::deep_search(const std::string& link_name) const {
	return pimpl_->deep_search<Key::Name>(link_name);
}

sp_link node::deep_search_oid(const std::string& oid) const {
	return pimpl_->deep_search<Key::OID>(oid);
}

BS_TYPE_IMPL(node, objbase, "node", "BS tree node", true, true);
BS_TYPE_ADD_CONSTRUCTOR(node, (std::string))
BS_REGISTER_TYPE("kernel", node)

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

