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
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/string_generator.hpp>

OMIT_OBJ_SERIALIZATION

NAMESPACE_BEGIN(blue_sky::tree)

const auto uuid_from_str = boost::uuids::string_generator{};

node_impl::node_impl(node* super)
	: timeout(def_timeout(true)), super_(super)
{}

// implement shallow links copy ctor
node_impl::node_impl(const node_impl& rhs, node* super)
	: timeout(rhs.timeout), super_(super)
{
	// [TODO] fix this in safer way after node is refactored
	auto& tgt = links_.get<Key_tag<Key::AnyOrder>>();
	for(const auto& plink : rhs.links_.get<Key_tag<Key::AnyOrder>>()) {
		tgt.insert(tgt.end(), plink.clone(true));
	}
	// [NOTE] links are in invalid state (no owner set) now
	// correct this by manually calling `node::propagate_owner()` after copy is constructed
}

node_impl::node_impl(node_impl&& rhs, node* super)
	: links_(std::move(rhs.links_)), handle_(std::move(rhs.handle_)),
	timeout(std::move(rhs.timeout)), actor_(std::move(rhs.actor_)), home_(std::move(rhs.home_)),
	super_(super)
{}

auto node_impl::spawn_actor(sp_nimpl nimpl, const std::string& gid) const -> caf::actor {
	// [NOTE] don't make shared_ptr here, because this can be called from node's ctor
	if(!super_) throw error{ "Can't spawn actor for null node" };
	return spawn_nactor(std::move(nimpl), gid);
}

auto node_impl::start_engine(sp_nimpl nimpl, std::string gid) -> void {
	if(!actor_)
		actor_ = spawn_actor(std::move(nimpl), std::move(gid));
	else // just enter group
		home(std::move(gid));
}

auto node_impl::home(std::string gid, bool silent) -> caf::group& {
	// generate random UUID for actor groud if passed GID is empty
	if(gid.empty()) {
		static auto uuid_gen = boost::uuids::random_generator{};
		gid = to_string(uuid_gen());
	}
	// make node's group & invite actor
	home_ = system().groups().get_local(gid);
	if(!silent)
		caf::anon_send<high_prio>(actor(), a_hi());
	return home_;
}

auto node_impl::super() const -> sp_node {
	return super_ ? super_->bs_shared_this<node>() : nullptr;
}

auto node_impl::set_handle(const link& new_handle) -> void {
	auto guard = lock<Metadata>();
	// remove node from existing owner if it differs from owner of new handle
	if(const auto old_handle = handle()) {
		const auto owner = old_handle.owner();
		if(owner && (!new_handle || owner != new_handle.owner()))
			owner->erase(old_handle.id());
	}

	// set new handle link
	handle_ = new_handle;
}

auto node_impl::handle() const -> link {
	return handle_.lock();
}

auto node_impl::size() const -> std::size_t {
	return links_.size();
}

auto node_impl::leafs(Key order) const -> links_v {
	switch(order) {
	case Key::AnyOrder:
		return values<Key::AnyOrder>();
	case Key::ID:
		return values<Key::ID>();
	case Key::Name:
		return values<Key::Name>();
	default:
		return {};
	}
}

auto node_impl::search(const std::string& key, Key key_meaning) const -> link {
	switch(key_meaning) {
	case Key::ID:
		return search<Key::ID>(uuid_from_str(key));
	case Key::Name:
		return search<Key::Name>(key);
	default:
		return {};
	}
}

auto node_impl::index(const std::string& key, Key key_meaning) const -> existing_index {
	switch(key_meaning) {
	case Key::ID:
		return index<Key::ID>(uuid_from_str(key));
	case Key::Name:
		return index<Key::Name>(key);
	default:
		return {};
	}
}

auto node_impl::equal_range(const std::string& key, Key key_meaning) const -> links_v {
	switch(key_meaning) {
	case Key::ID:
		return equal_range<Key::ID>(uuid_from_str(key)).extract_values();
	case Key::Name:
		return equal_range<Key::Name>(key).extract_values();
	default:
		return {};
	}
}

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
// hardcode number of rename trials on insertion
inline constexpr auto rename_trials = 10000;

auto node_impl::insert(
	link L, const InsertPolicy pol, leaf_postproc_fn ppf
) -> insert_status<Key::ID> {
	// can't move persistent node from it's owner
	const auto Lflags = L.flags();
	if(!L || (Lflags & Flags::Persistent && L.owner()))
		return { {}, false };

	// 1. check if we have duplicated name and have to rename link after insertion
	// If insert policy deny duplicating name & we have to rename link being inserted
	// then new link name will go here
	auto Lname = std::optional<std::string>{};
	if(enumval(pol & 3) > 0) {
		const auto old_name = L.name();
		auto dup = find<Key::Name, Key::ID>(old_name);
		if(dup != end<Key::ID>() && dup->id() != L.id()) {
			// first check if dup names are prohibited
			if(enumval(pol & InsertPolicy::DenyDupNames) || Lflags & Flags::Persistent)
				return { std::move(dup), false };
			else if(enumval(pol & InsertPolicy::RenameDup)) {
				// try to auto-rename link
				auto names_end = end<Key::Name>();
				for(int i = 0; i < rename_trials; ++i) {
					auto new_name = old_name + '_' + std::to_string(i);
					if(find<Key::Name, Key::Name>(new_name) == names_end) {
						Lname = std::move(new_name);
						break;
					}
				}
			}
			// if no unique name was found - return fail
			if(!Lname) return { std::move(dup), false };
		}
	}

	// 2. make insertion
	// [NOTE] reset link's owner to insert safely (block symlink side effects, etc)
	const auto prev_owner = L.owner();
	L.reset_owner(nullptr);
	auto res = links_.get<Key_tag<Key::ID>>().insert(L);

	// 3. postprocess
	auto& res_L = *res.first;
	if(res.second) {
		// remove from prev parent and propagate handle while link's owner still NULL (important!)
		const auto self = super();
		if(prev_owner && prev_owner != self)
			// erase won't touch owner (we set it manually)
			caf::anon_send(
				node_impl::actor(*prev_owner), a_node_erase(), res_L.id(), EraseOpts::DontResetOwner
			);
		res_L.propagate_handle();
		// set owner to this node
		res_L.reset_owner(self);

		// rename link if needed
		if(Lname) res_L.rename(std::move(*Lname));
		// invoke postprocessing of just inserted link
		ppf(res_L);
	}
	else {
		// restore link's original owner
		L.reset_owner(prev_owner);
		// check if we need to deep merge given links
		// go one step down the hierarchy
		if(enumval(pol & InsertPolicy::Merge) && res.first != end<Key::ID>()) {
			auto src_node = L.data_node();
			auto dst_node = res_L.data_node();
			if(src_node && dst_node) {
				// insert all links from source node into destination
				dst_node->insert(src_node->leafs(), pol);
			}
		}
	}
	return res;
}

// postprocessing of just inserted link
// if link points to node, return it
auto node_impl::adjust_inserted_link(const link& lnk, const sp_node& target) -> sp_node {
	// sanity
	if(!lnk) return nullptr;

	// change link's owner
	if(auto prev_owner = lnk.owner(); prev_owner != target) {
		lnk.reset_owner(target);
		// [NOTE] instruct prev node to not reset link's owner - we set above
		if(prev_owner)
			caf::anon_send(
				node_impl::actor(*prev_owner), a_node_erase(), lnk.id(), EraseOpts::DontResetOwner
			);
	}

	// if we're inserting a node, relink it to ensure a single hard link exists
	return lnk.propagate_handle().value_or(nullptr);
}

auto node_impl::erase_impl(
	iterator<Key::ID> victim, leaf_postproc_fn ppf, bool dont_reset_owner
) -> std::size_t {
	// preprocess before erasing
	auto L = *victim;
	ppf(L);

	// erase
	if(!dont_reset_owner) L.reset_owner(nullptr);
	auto res = to_index<Key::ID>(victim);
	links_.get<Key_tag<Key::ID>>().erase(victim);
	return res.value_or(links_.size());
}

auto node_impl::erase(const std::string& key, Key key_meaning, leaf_postproc_fn ppf) -> size_t {
	switch(key_meaning) {
	case Key::ID:
		return erase<Key::ID>(uuid_from_str(key), ppf);
	case Key::Name:
		return erase<Key::Name>(key, ppf);
	default:
		return 0;
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
auto node_impl::refresh(const lid_type& lid) -> void {
	constexpr auto touch = [](auto&&) {};

	// find target link by it's ID
	auto& I = links_.get<Key_tag<Key::ID>>();
	if(auto pos = I.find(lid); pos != I.end()) {
		// refresh each index in cycle
		links_.get<Key_tag<Key::Name>>().modify_key( project<Key::ID, Key::Name>(pos), touch );
	}
}

NAMESPACE_END(blue_sky::tree)
