/// @file
/// @author uentity
/// @date 29.06.2018
/// @brief BS tree node implementation part of PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/node.h>
#include <bs/kernel/config.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/boost_uuid.h>

#include "actor_common.h"

#include <set>
#include <mutex>
#include <unordered_map>

#include <boost/uuid/uuid_hash.hpp>
#include <caf/event_based_actor.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::config;

using links_container = node::links_container;
using Key = node::Key;
template<Key K> using iterator = typename node::iterator<K>;
template<Key K> using Key_tag = typename node::Key_tag<K>;
template<Key K> using Key_type = typename node::Key_type<K>;
template<Key K> using Key_const = typename node::Key_const<K>;
template<Key K> using insert_status = typename node::insert_status<K>;
template<Key K> using range = typename node::range<K>;

using Flags = link::Flags;
using Req = link::Req;
using ReqStatus = link::ReqStatus;
using InsertPolicy = node::InsertPolicy;

/*-----------------------------------------------------------------------------
 *  node_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API node_actor : public caf::event_based_actor {
public:
	friend struct access_node_actor;
	using super = caf::event_based_actor;

	// timeout for most queries
	timespan timeout_;

	std::weak_ptr<link> handle_;
	links_container links_;
	std::vector<std::string> allowed_otypes_;
	// temp guard until caf-based tree implementation is ready
	std::mutex links_guard_;
	caf::group self_grp;

	// map links to retranslator actors
	std::unordered_map<link::id_type, std::uint64_t> lnk_wires_;

	// default ctor
	node_actor(caf::actor_config& cfg, const std::string& id, timespan data_timeout = def_data_timeout);

	// performs deep links copy
	node_actor(caf::actor_config& cfg, const std::string& id, caf::actor src);

	// say goodbye to others & leave self group
	auto goodbye() -> void;

	auto make_behavior() -> behavior_type override;

	template<Key K = Key::ID>
	static auto deep_search_impl(
		const node_actor& n, const Key_type<K>& key,
		std::set<Key_type<Key::ID>> active_symlinks = {}
	) -> sp_link {
		// first do direct search in leafs
		auto r = n.find<K, K>(key);
		if(r != n.end<K>()) return *r;

		// if not succeeded search in children nodes
		for(const auto& l : n.links_) {
			// remember symlink
			const auto is_symlink = l->type_id() == "sym_link";
			if(is_symlink){
				if(active_symlinks.find(l->id()) == active_symlinks.end())
					active_symlinks.insert(l->id());
				else continue;
			}
			// check populated status before moving to next level
			if(l->flags() & link::LazyLoad && l->req_status(Req::DataNode) != ReqStatus::OK)
				continue;
			// search on next level
			if(const auto next_n = l->data_node()) {
				auto next_l = deep_search_impl<K>(*next_n->pimpl_, key, active_symlinks);
				if(next_l) return next_l;
			}
			// remove symlink
			if(is_symlink)
				active_symlinks.erase(l->id());
		}
		return nullptr;
	}

	template<Key K = Key::ID>
	auto deep_search(const Key_type<K>& key) const -> sp_link {
		return this->deep_search_impl<K>(*this, key);
	}

	template<Key K = Key::AnyOrder>
	auto begin() const {
		return links_.get<Key_tag<K>>().begin();
	}
	template<Key K = Key::AnyOrder>
	auto end() const {
		return links_.get<Key_tag<K>>().end();
	}

	template<Key K>
	auto project(iterator<K> pos) const {
		return links_.project<Key_tag<Key::AnyOrder>>(std::move(pos));
	}

	template<Key K>
	auto keys() const -> std::vector<Key_type<K>> {
		std::set<Key_type<K>> r;
		auto kex = Key_tag<K>();
		for(const auto& i : links_)
			r.insert(kex(*i));
		return {r.begin(), r.end()};
	}

	template<Key K, Key R = Key::AnyOrder>
	auto find(const Key_type<K>& key) const {
		auto res = links_.get<Key_tag<K>>().find(key);
		if constexpr(std::is_same_v<Key_const<K>, Key_const<R>>)
			return res;
		else
			return links_.project<Key_tag<R>>( std::move(res) );
	}

	template<Key K = Key::ID>
	auto equal_range(const Key_type<K>& key) const {
		return links_.get<Key_tag<K>>().equal_range(key);
	}

	template<Key K = Key::ID>
	auto erase(const Key_type<K>& key) -> void {
		auto solo = std::lock_guard{ links_guard_ };
		auto victim = find<K, Key::ID>(key);

		if(victim != end<Key::ID>()) {
			// stop retranslator
			const auto lid = (*victim)->id();
			const auto rid = lnk_wires_[lid];
			auto& Reg = actor_system().registry();
			send(caf::actor_cast<caf::actor>(Reg.get( rid )), a_bye());
			// send message that link erased
			send(self_grp, a_lnk_erase(), lid);

			// and erase link
			links_.get<Key_tag<Key::ID>>().erase(victim);
		}
	}

	template<Key K = Key::ID>
	auto erase(const range<K>& r) -> void {
		//auto solo = std::lock_guard{ links_guard_ };
		for(const auto& k : r)
			erase<K>(links_.get<Key_tag<K>>().key_extractor()(k));
		//links_.get<Key_tag<K>>().erase(r.first, r.second);
	}

	auto insert(sp_link L, const InsertPolicy pol) -> insert_status<Key::ID>;

	template<Key K>
	bool rename(iterator<K>&& pos, std::string&& new_name) {
		auto solo = std::lock_guard{ links_guard_ };

		if(pos == end<K>()) return false;
		return links_.get<Key_tag<K>>().modify(pos, [name = std::move(new_name)](sp_link& l) {
			l->rename_silent(std::move(name));
		});
	}

	template<Key K>
	std::size_t rename(const Key_type<K>& key, const std::string& new_name, bool all = false) {
		auto solo = std::lock_guard{ links_guard_ };

		range<K> matched_items = equal_range<K>(key);
		auto& storage = links_.get<Key_tag<K>>();
		auto renamer = [&new_name](sp_link& l) {
			l->rename_silent(new_name);
		};
		int cnt = 0;
		for(auto pos = matched_items.begin(); pos != matched_items.end(); ++pos) {
			storage.modify(pos, renamer);
			++cnt;
			if(!all) break;
		}
		return cnt;
	}

	void on_rename(const Key_type<Key::ID>& key);

	bool accepts(const sp_link& what) const;

	void set_handle(const sp_link& new_handle);

	// postprocessing of just inserted link
	// if link points to node, return it
	static sp_node adjust_inserted_link(const sp_link& lnk, const sp_node& n);
};

NAMESPACE_END(blue_sky::tree)
