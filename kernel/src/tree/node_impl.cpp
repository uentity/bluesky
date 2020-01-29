/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Implementataion of node actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_impl.h"
#include "link_impl.h"
#include "node_actor.h"
#include <bs/log.h>
#include <bs/tree/tree.h>
#include <bs/detail/tuple_utils.h>
#include <bs/serialize/cafbind.h>

#include <boost/uuid/uuid_io.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

const auto uuid_from_str = boost::uuids::string_generator{};

node_impl::node_impl(node* super)
	: timeout(def_timeout(true)), factor(kernel::radio::system()), super_(super)
{}

// implement shallow links copy ctor
node_impl::node_impl(const node_impl& rhs, node* super)
	: allowed_otypes_(rhs.allowed_otypes_),
	timeout(rhs.timeout), factor(kernel::radio::system()), super_(super)
{
	for(const auto& plink : rhs.links_.get<Key_tag<Key::AnyOrder>>()) {
		// non-deep clone can result in unconditional moving nodes from source
		insert(plink->clone(true), InsertPolicy::AllowDupNames);
	}
	// [NOTE] links are in invalid state (no owner set) now
	// correct this by manually calling `node::propagate_owner()` after copy is constructed
}

node_impl::node_impl(node_impl&& rhs, node* super)
	: links_(std::move(rhs.links_)),
	handle_(std::move(rhs.handle_)), allowed_otypes_(std::move(rhs.allowed_otypes_)),
	timeout(std::move(rhs.timeout)), factor(kernel::radio::system()), self_grp(std::move(rhs.self_grp)),
	super_(super)
{}

auto node_impl::spawn_actor(std::shared_ptr<node_impl> nimpl, const std::string& gid) const -> caf::actor {
	// [NOTE] don't make shared_ptr here, because this can be called from node's ctor
	if(!super_) throw error{ "Can't spawn actor for null node" };
	return spawn_nactor(std::move(nimpl), gid);
}

auto node_impl::gid() const -> std::string {
	// `self_grp` cannot change after being set
	return self_grp ? self_grp.get()->identifier() : "";
}

auto node_impl::super() const -> sp_node {
	return super_ ? super_->bs_shared_this<node>() : nullptr;
}

auto node_impl::set_handle(const sp_link& new_handle) -> void {
	auto guard = lock<Metadata>();
	// remove node from existing owner if it differs from owner of new handle
	if(const auto old_handle = handle_.lock()) {
		const auto owner = old_handle->owner();
		if(owner && (!new_handle || owner != new_handle->owner()))
			owner->erase(old_handle->id());
	}

	// set new handle link
	handle_ = new_handle;
}

auto node_impl::propagate_owner(bool deep) -> void {
	// properly setup owner in node's leafs
	for(auto& plink : links_) {
		auto child_node = adjust_inserted_link(plink, super());
		if(deep && child_node)
			child_node->propagate_owner(true);
	}
}

auto node_impl::size() const -> std::size_t {
	return links_.size();
}

auto node_impl::leafs(Key order) const -> links_v {
	switch(order) {
	default:
	case Key::AnyOrder:
		return values<Key::AnyOrder>();
	case Key::ID:
		return values<Key::ID>();
	case Key::Name:
		return values<Key::Name>();
	case Key::OID:
		return values<Key::OID>();
	case Key::Type:
		return values<Key::Type>();
	}
}

auto node_impl::search(const std::string& key, Key key_meaning) const -> sp_link {
	switch(key_meaning) {
	case Key::ID:
		return search<Key::ID>(uuid_from_str(key));
	case Key::OID:
		return search<Key::OID>(key);
	case Key::Type:
		return search<Key::Type>(key);
	default:
	case Key::Name:
		return search<Key::Name>(key);
	}
}

auto node_impl::deep_search(const std::string& key, Key key_meaning) const -> sp_link {
	switch(key_meaning) {
	case Key::ID:
		return deep_search<Key::ID>(uuid_from_str(key));
	case Key::OID:
		return deep_search<Key::OID>(key);
	case Key::Type:
		return deep_search<Key::Type>(key);
	default:
	case Key::Name:
		return deep_search<Key::Name>(key);
	}
}

auto node_impl::index(const std::string& key, Key key_meaning) const -> existing_index {
	switch(key_meaning) {
	case Key::ID:
		return index<Key::ID>(uuid_from_str(key));
	case Key::OID:
		return index<Key::OID>(key);
	case Key::Type:
		return index<Key::Type>(key);
	default:
	case Key::Name:
		return index<Key::Name>(key);
	}
}

auto node_impl::equal_range(const std::string& key, Key key_meaning) const -> links_v {
	switch(key_meaning) {
	case Key::ID:
		return equal_range<Key::ID>(uuid_from_str(key)).extract_values();
	case Key::OID:
		return equal_range<Key::OID>(key).extract_values();
	case Key::Type:
		return equal_range<Key::Type>(key).extract_values();
	default:
	case Key::Name:
		return equal_range<Key::Name>(key).extract_values();
	}
}

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
// hardcode number of rename trials on insertion
inline constexpr auto rename_trials = 10000;

auto node_impl::insert(
	sp_link L, const InsertPolicy pol, leaf_postproc_fn ppf
) -> insert_status<Key::ID> {
	// can't move persistent node from it's owner
	const auto Lflags = L->flags();
	if(!L || !accepts(L) || (Lflags & Flags::Persistent && L->owner()))
		return { {}, false };

	// if insert policy deny duplicating name & we have to rename link being inserted
	// then new link name will go here
	auto Lname = std::optional<std::string>{};

	// setup postprocessing step after successfull or failed insertion
	const auto make_result = [&, prev_owner = L->owner()](insert_status<Key::ID>&& res) {
		// work with link's original owner depending on if insert happened or not
		auto& res_L = *res.first;
		if(res.second) {
			if(prev_owner && prev_owner != res_L->owner())
				// [NOTE] on successfull insertion valid owner is already set
				caf::anon_send(
					node_impl::actor(*prev_owner), a_node_erase(), res_L->id(), EraseOpts::DontResetOwner
				);
			// set handle if link contains node
			res_L->propagate_handle();
			// rename link if needed
			if(Lname) res_L->rename(std::move(*Lname));
			// invoke postprocessing of just inserted link
			ppf(res_L);
		}
		else {
			// restore link's original owner
			L->reset_owner(std::move(prev_owner));
			// check if we need to deep merge given links
			// go one step down the hierarchy
			if(enumval(pol & InsertPolicy::Merge) && res.first != end<Key::ID>()) {
				// check if we need to deep merge given links
				// go one step down the hierarchy
				auto src_node = L->data_node();
				auto dst_node = res_L->data_node();
				if(src_node && dst_node) {
					// insert all links from source node into destination
					dst_node->insert(src_node->leafs(), pol);
				}
			}
		}

		return std::move(res);
	};

	// ----- insertion starts here
	// 1. check if we have duplicated name and have to rename link after insertion
	iterator<Key::ID> dup;
	if(enumval(pol & 3) > 0) {
		const auto old_name = L->name();
		dup = find<Key::Name, Key::ID>(old_name);
		if(dup != end<Key::ID>() && (*dup)->id() != L->id()) {
			// first check if dup names are prohibited
			if(enumval(pol & InsertPolicy::DenyDupNames) || Lflags & Flags::Persistent)
				return {dup, false};
			else if(enumval(pol & InsertPolicy::RenameDup)) {
				// try to auto-rename link
				for(int i = 0; i < rename_trials; ++i) {
					auto new_name = old_name + '_' + std::to_string(i);
					if(find<Key::Name, Key::Name>(new_name) == end<Key::Name>()) {
						Lname = std::move(new_name);
						break;
					}
				}
			}
			// if no unique name was found - return fail
			if(!Lname) return {dup, false};
		}
	}

	// 2. early setup new owner, because sym links can return proper OID only if owner is valid
	if(const auto& target = super()) L->reset_owner(target);

	// 3. check for duplicating OID
	auto& I = links_.get<Key_tag<Key::ID>>();
	if( enumval(pol & (InsertPolicy::DenyDupOID | InsertPolicy::ReplaceDupOID)) ) {
		if(dup = find<Key::OID, Key::ID>(L->oid()); dup != end<Key::ID>()) {
			bool is_inserted = false;
			if(enumval(pol & InsertPolicy::ReplaceDupOID))
				is_inserted = I.replace(dup, std::move(L));
			return make_result({ dup, is_inserted });
		}
	}

	// 4. insert & postprocess
	return make_result(I.insert(std::move(L)));
}

// postprocessing of just inserted link
// if link points to node, return it
auto node_impl::adjust_inserted_link(const sp_link& lnk, const sp_node& target) -> sp_node {
	// sanity
	if(!lnk) return nullptr;

	// change link's owner
	// [NOTE] for sym links it's important to set new owner early
	// otherwise, `data()` will return null and statuses will become Error
	auto prev_owner = lnk->owner();
	if(prev_owner != target) {
		if(prev_owner) prev_owner->erase(lnk->id());
		lnk->reset_owner(target);
	}

	// if we're inserting a node, relink it to ensure a single hard link exists
	return lnk->propagate_handle().value_or(nullptr);
}

auto node_impl::erase_impl(
	iterator<Key::ID> victim, leaf_postproc_fn ppf, bool dont_reset_owner
) -> std::size_t {
	// preprocess before erasing
	auto L = *victim;
	ppf(L);
	// erase
	if(!dont_reset_owner) L->reset_owner(nullptr);
	auto& I = links_.get<Key_tag<Key::ID>>();
	auto res = I.erase(victim);
	return static_cast<std::size_t>(std::distance(I.begin(), res));
}

auto node_impl::erase(const std::string& key, Key key_meaning, leaf_postproc_fn ppf) -> size_t {
	switch(key_meaning) {
	case Key::ID:
		return erase<Key::ID>(uuid_from_str(key), ppf);
	case Key::OID:
		return erase<Key::OID>(key, ppf);
	case Key::Type:
		return erase<Key::Type>(key, ppf);
	default:
	case Key::Name:
		return erase<Key::Name>(key, ppf);
	}
}

auto node_impl::erase(const lids_v& r, leaf_postproc_fn ppf) -> std::size_t {
	std::size_t res = 0;
	std::for_each(
		r.begin(), r.end(),
		[&](const auto& lid) { res += erase(lid, ppf); }
	);
	return res;
}

///////////////////////////////////////////////////////////////////////////////
//  misc
//
auto node_impl::rename(
	const std::string& key, const std::string& new_name, Key key_meaning, bool all
) -> size_t {
	switch(key_meaning) {
	case Key::ID:
		return rename<Key::ID>(uuid_from_str(key), new_name, all);
	case Key::OID:
		return rename<Key::OID>(key, new_name, all);
	case Key::Type:
		return rename<Key::Type>(key, new_name, all);
	default:
	case Key::Name:
		return rename<Key::Name>(key, new_name, all);
	}
}

auto node_impl::refresh(const lid_type& lid) -> void {
	constexpr auto touch = [](auto&&) {};

	// find target link by it's ID
	auto& I = links_.get<Key_tag<Key::ID>>();
	if(auto pos = I.find(lid); pos != I.end()) {
		// refresh each index in cycle
		links_.get<Key_tag<Key::Name>>().modify_key( project<Key::ID, Key::Name>(pos), touch );
		links_.get<Key_tag<Key::OID> >().modify_key( project<Key::ID, Key::OID>(pos),  touch );
		links_.get<Key_tag<Key::Type>>().modify_key( project<Key::ID, Key::Type>(pos), touch );
	}
}

auto node_impl::accepts(const sp_link& what) const -> bool {
	if(!allowed_otypes_.size()) return true;
	const auto& what_type = what->obj_type_id();
	for(const auto& otype : allowed_otypes_) {
		if(what_type == otype) return true;
	}
	return false;
}

auto node_impl::accept_object_types(std::vector<std::string> allowed_types) -> void {
	auto guard = lock<Metadata>();
	allowed_otypes_ = std::move(allowed_types);
}

NAMESPACE_END(blue_sky::tree)
