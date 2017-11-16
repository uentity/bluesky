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
#include <set>

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

using Flags = link::Flags;

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
		for(const auto& l : n.links_) {
			if(auto n = l->data_node()) {
				auto l = deep_search_impl<K>(*n->pimpl_, key);
				if(l) return l;
			}
		}
		return nullptr;
	}

	//static deep_merge(const node_impl& n, )

	template<Key K = Key::AnyOrder>
	auto begin() const {
		return links_.get<Key_tag<K>>().begin();
	}
	template<Key K = Key::AnyOrder>
	auto end() const {
		return links_.get<Key_tag<K>>().end();
	}

	template<Key K, Key R = Key::AnyOrder>
	auto find(
		const Key_type<K>& key,
		std::enable_if_t<std::is_same<Key_const<K>, Key_const<R>>::value>* = nullptr
	) const {
		return links_.get<Key_tag<K>>().find(key);
	}

	template<Key K, Key R = Key::AnyOrder>
	auto find(
		const Key_type<K>& key,
		std::enable_if_t<!std::is_same<Key_const<K>, Key_const<R>>::value>* = nullptr
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

	insert_status<Key::ID> insert(sp_link l, InsertPolicy pol) {
		// can't move persistent node from it's owner
		if(!l || !accepts(l) || (l->flags() & Flags::Persistent && l->owner()))
			return {end<Key::ID>(), false};
		// check if we have duplication name
		iterator<Key::ID> dup;
		if((pol & 3) > 0) {
			dup = find<Key::Name, Key::ID>(l->name());
			if(dup != end<Key::ID>()) {
				bool unique_found = false;
				// first check if dup names are prohibited
				if(pol & InsertPolicy::DenyDupNames) return {dup, false};
				else if(pol & InsertPolicy::RenameDup && !(l->flags() & Flags::Persistent)) {
					// try to auto-rename link
					std::string new_name;
					for(int i = 0; i < 10000; ++i) {
						new_name = l->name() + '_' + std::to_string(i);
						if(find<Key::Name, Key::Name>(new_name) == end<Key::Name>()) {
							// we've found a unique name
							l->name_ = std::move(new_name);
							unique_found = true;
							break;
						}
					}
				}
			// if no unique name was found - return fail
			if(!unique_found) return {dup, false};
			}
		}
		// check for duplicating OID
		if(pol & InsertPolicy::DenyDupOID) {
			dup = find<Key::OID, Key::ID>(l->oid());
			if(dup != end<Key::ID>()) return {dup, false};
		}
		// try to insert given link
		return links_.get<Key_tag<Key::ID>>().insert(std::move(l));
	}

	template<Key K>
	std::vector<Key_type<K>> keys() const {
		std::set<Key_type<K>> r;
		auto kex = Key_tag<K>();
		for(const auto& i : links_)
			r.insert(kex(*i));
		return {r.begin(), r.end()};
	}

	template<Key K>
	bool rename(iterator<K> pos, std::string new_name) {
		if(pos == end<K>()) return false;
		return links_.get<Key_tag<K>>().modify(pos, [name = std::move(new_name)](sp_link& l) {
			l->name_ = std::move(name);
		});
	}

	template<Key K>
	auto project(iterator<K> pos) const {
		return links_.project<Key_tag<Key::AnyOrder>>(std::move(pos));
	}

	bool accepts(const sp_link& what) const {
		if(!allowed_otypes_.size()) return true;
		const auto& what_type = what->obj_type_id();
		for(const auto& otype : allowed_otypes_) {
			if(what_type == otype) return true;
		}
		return false;
	}

	// implement deep copy ctor
	node_impl(const node_impl& src) {
		for(const auto& plink : src.links_) {
			insert(plink->clone(), InsertPolicy::AllowDupNames);
			//links_.get<Key_tag<Key::ID>>().insert(plink->clone());
		}
	}

	node_impl() = default;

	links_container links_;
	std::vector<std::string> allowed_otypes_;
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

iterator<Key::AnyOrder> node::find(const std::string& name) const {
	return pimpl_->find<Key::Name>(name);
}

iterator<Key::AnyOrder> node::find_oid(const std::string& oid) const {
	return pimpl_->find<Key::OID>(oid);
}

// ---- index
std::size_t node::index(const id_type& lid) const {
	auto i = pimpl_->find<Key::ID, Key::AnyOrder>(lid);
	return std::distance(begin<Key::AnyOrder>(), i);
}

std::size_t node::index(const iterator<Key::AnyOrder>& pos) const {
	return std::distance(begin<Key::AnyOrder>(), pos);
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
insert_status<Key::ID> node::insert(const sp_link& l, InsertPolicy pol) {
	auto res = pimpl_->insert(l, pol);
	// switch owner to *this if succeeded
	if(res.second) {
		auto& res_lnk = *(res.first->get());
		if(auto prev_owner = res_lnk.owner())
			prev_owner->erase(res_lnk.id());
		res_lnk.reset_owner(bs_shared_this<node>());
	}
	else if(pol & InsertPolicy::Merge && res.first != end<Key::ID>()) {
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

insert_status<Key::AnyOrder> node::insert(const sp_link& l, iterator<> pos, InsertPolicy pol) {
	// 1. insert an element using ID index
	auto res = insert(l, pol);
	if(res.first != end<Key::ID>()) {
		// 2. reposition an element in AnyOrder index
		auto src = pimpl_->project<Key::ID>(res.first);
		if(pos != src) {
			auto& ord_idx = pimpl_->links_.get<Key_tag<Key::AnyOrder>>();
			ord_idx.relocate(pos, src);
			return {src, res.second};
		}
	}
	return {pimpl_->project<Key::ID>(res.first), res.second};
}

insert_status<Key::AnyOrder> node::insert(const sp_link& l, std::size_t idx, InsertPolicy pol) {
	return insert(l, std::next(begin<Key::AnyOrder>(), std::min(idx, size())), pol);
}

insert_status<Key::ID> node::insert(std::string name, sp_obj obj, InsertPolicy pol) {
	return insert(
		std::make_shared<hard_link>(std::move(name), std::move(obj)), pol
	);
}

// ---- erase
void node::erase(const std::size_t idx) {
	pimpl_->links_.get<Key_tag<Key::AnyOrder>>().erase(find(idx));
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

void node::erase_type(const std::string& type_id) {
	pimpl_->erase<Key::Type>(type_id);
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

// ---- deep_search
sp_link node::deep_search(const id_type& id) const {
	return pimpl_->deep_search<>(id);
}

sp_link node::deep_search(const std::string& link_name) const {
	return pimpl_->deep_search<Key::Name>(link_name);
}

sp_link node::deep_search_oid(const std::string& oid) const {
	return pimpl_->deep_search<Key::OID>(oid);
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

bool node::rename(iterator<Key::AnyOrder> pos, std::string new_name) {
	return pimpl_->rename<Key::AnyOrder>(std::move(pos), std::move(new_name));
}

iterator<Key::AnyOrder> node::project(iterator<Key::ID> src) const {
	return pimpl_->project<Key::ID>(src);
}

iterator<Key::AnyOrder> node::project(iterator<Key::Name> src) const {
	return pimpl_->project<Key::Name>(src);
}

iterator<Key::AnyOrder> node::project(iterator<Key::OID> src) const {
	return pimpl_->project<Key::OID>(src);
}

iterator<Key::AnyOrder> node::project(iterator<Key::Type> src) const {
	return pimpl_->project<Key::Type>(src);
}

bool node::accepts(const sp_link& what) const {
	return pimpl_->accepts(what);
}

void node::accept_object_types(std::vector<std::string> allowed_types) {
	pimpl_->allowed_otypes_ = std::move(allowed_types);
}

std::vector<std::string> node::allowed_object_types() const {
	return pimpl_->allowed_otypes_;
}

sp_node node::deep_clone(InsertPolicy pol) const {
	sp_node res = std::make_shared<node>();
	for(const auto& plink : pimpl_->links_) {
		res->insert(plink->clone(true), pol);
	}
	return res;
}

BS_TYPE_IMPL(node, objbase, "node", "BS tree node", true, true);
BS_TYPE_ADD_CONSTRUCTOR(node, (std::string))
BS_REGISTER_TYPE("kernel", node)

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

