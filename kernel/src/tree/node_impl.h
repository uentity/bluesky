/// @file
/// @author uentity
/// @date 29.06.2018
/// @brief BS tree node implementation part of PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/actor_common.h>
#include <bs/log.h>
#include <bs/tree/node.h>
#include <bs/detail/function_view.h>
#include <bs/detail/sharded_mutex.h>

#include <boost/uuid/uuid_hash.hpp>

#include <cereal/types/vector.hpp>

#include <set>
#include <unordered_map>
#include <algorithm>

NAMESPACE_BEGIN(blue_sky::tree)
namespace bs_detail = blue_sky::detail;

using id_type = link_id_type;
using links_container = node::links_container;
using EraseOpts = node::EraseOpts;

template<Key K> using iterator = typename node::iterator<K>;
template<Key K> using Key_tag = typename node::Key_tag<K>;
template<Key K> using Key_type = typename node::Key_type<K>;
template<Key K> using Key_const = typename node::Key_const<K>;
template<Key K> using insert_status = typename node::insert_status<K>;
template<Key K> using range = typename node::range<K>;
template<Key K> using const_range = typename node::const_range<K>;

using bs_detail::shared;
using node_impl_mutex = std::shared_mutex;

static constexpr auto noop_postproc_f = [](const auto&) {};

/*-----------------------------------------------------------------------------
 *  node_impl
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API node_impl : public bs_detail::sharded_same_mutex<node_impl_mutex, 2> {
public:
	friend class node;

	// lock granularity
	enum { Metadata, Links };

	// leafs
	links_container links_;
	// metadata
	std::weak_ptr<link> handle_;
	std::vector<std::string> allowed_otypes_;

	// timeout for most queries
	const caf::duration timeout;
	// scoped actor for requests
	caf::scoped_actor factor;

	// local node group
	caf::group self_grp;

	// append private behavior to public iface
	using actor_type = node::actor_type::extend<
        // terminate actor
		caf::reacts_to<a_bye>,
		// track link rename
		caf::reacts_to<a_ack, a_lnk_rename, id_type, std::string, std::string>,
		// track link status
		caf::reacts_to<a_ack, a_lnk_status, id_type, Req, ReqStatus, ReqStatus>,
		// ack on insert - reflect insert from sibling node actor
		caf::reacts_to<a_ack, a_lnk_insert, id_type, size_t, InsertPolicy>,
		// ack on link move
		caf::reacts_to<a_ack, a_lnk_insert, id_type, size_t, size_t>,
		// ack on link erase from sibling node
		caf::reacts_to<a_ack, a_lnk_erase, std::vector<id_type>, std::vector<std::string>>
	>;

	static auto actor(const node& N) {
		return caf::actor_cast<actor_type>(N.actor_);
	}

	// make request to given link L
	template<typename R, typename... Args>
	auto actorf(const node& N, Args&&... args) {
		return blue_sky::actorf<R>(
			factor, caf::actor_cast<actor_type>(N.actor_), timeout, std::forward<Args>(args)...
		);
	}
	// same as above but with configurable timeout
	template<typename R, typename... Args>
	auto actorf(const node& N, timespan timeout, Args&&... args) {
		return blue_sky::actorf<R>(
			factor, caf::actor_cast<actor_type>(N.actor_), timeout, std::forward<Args>(args)...
		);
	}

	virtual auto spawn_actor(std::shared_ptr<node_impl> nimpl, const std::string& gid) const -> caf::actor;

	// default & copy ctor
	node_impl(node* super);
	node_impl(const node_impl&, node* super);
	node_impl(node_impl&&, node* super);

	///////////////////////////////////////////////////////////////////////////////
	//  search
	//
	template<Key K = Key::ID>
	static auto deep_search_impl(
		const node_impl& n, const Key_type<K>& key,
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
			if(l->flags() & LazyLoad && l->req_status(Req::DataNode) != ReqStatus::OK)
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

	template<Key K, Key R = Key::AnyOrder>
	auto find(const Key_type<K>& key) const {
		if constexpr(K == Key::AnyOrder)
			return std::next(begin<K>(), key);
		else
			return project<K, R>(links_.get<Key_tag<K>>().find(key));
	}

	template<Key K = Key::ID>
	auto equal_range(const Key_type<K>& key) const -> range<K> {
		if constexpr(K != Key::AnyOrder) {
			return links_.get<Key_tag<K>>().equal_range(key);
		}
		else {
			auto pos = find<K>(key);
			return {pos, std::next(pos)};
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	//  iterate
	//
	template<Key K = Key::AnyOrder>
	auto begin() const {
		return links_.get<Key_tag<K>>().begin();
	}
	template<Key K = Key::AnyOrder>
	auto end() const {
		return links_.get<Key_tag<K>>().end();
	}

	template<Key K, Key R = Key::AnyOrder>
	auto project(iterator<K> pos) const {
		if constexpr(K == R)
			return pos;
		else
			return links_.project<Key_tag<R>>(std::move(pos));
	}

	template<Key K>
	auto keys() const -> std::vector<Key_type<K>> {
		auto kex = Key_tag<K>();
		auto res = std::vector<Key_type<K>>(links_.size());
		std::transform(
			links_.begin(), links_.end(), res.begin(),
			[&](const auto& L) { return kex(*L); }
		);
		std::sort(res.begin(), res.end());
		return res;
	}

	template<Key K>
	auto values() const -> links_v {
		auto res = links_v(links_.size());
		std::copy(begin<K>(), end<K>(), res.begin());
		return res;
	}

	auto leafs(Key order) const -> links_v;

	auto size() const -> std::size_t;

	///////////////////////////////////////////////////////////////////////////////
	//  insert
	//
	using leaf_postproc_fn = function_view< void(const sp_link&) >;

	auto insert(
		sp_link L, const InsertPolicy pol = InsertPolicy::AllowDupNames,
		leaf_postproc_fn ppf = noop_postproc_f
	) -> insert_status<Key::ID>;

	///////////////////////////////////////////////////////////////////////////////
	//  erase
	//
	template<Key K = Key::ID>
	auto erase(
		const range<K>& r, leaf_postproc_fn ppf = noop_postproc_f,
		bool dont_reset_owner = false
	) -> size_t {
		size_t res = 0;
		for(auto x = r.begin(); x != r.end();) {
			if constexpr(K == Key::ID)
				erase_impl(x++, ppf, dont_reset_owner);
			else
				erase_impl(project<K, Key::ID>(x++), ppf, dont_reset_owner);
			++res;
		}
		return res;
	}

	template<Key K = Key::ID>
	auto erase(
		const Key_type<K>& key, leaf_postproc_fn ppf = noop_postproc_f,
		bool dont_reset_owner = false
	) -> size_t {
		return erase<K>(equal_range<K>(key), ppf, dont_reset_owner);
	}

	///////////////////////////////////////////////////////////////////////////////
	//  rename
	//
	template<Key K>
	bool rename(iterator<K>&& pos, std::string&& new_name) {
		if(pos == end<K>()) return false;
		return links_.get<Key_tag<K>>().modify(pos, [name = std::move(new_name)](sp_link& l) {
			l->rename(std::move(name));
		});
	}

	template<Key K>
	std::size_t rename(const Key_type<K>& key, const std::string& new_name, bool all = false) {
		auto matched_items = equal_range<K>(key);
		auto& storage = links_.get<Key_tag<K>>();
		auto renamer = [&new_name](sp_link& l) {
			l->rename(new_name);
		};
		int cnt = 0;
		for(auto pos = matched_items.begin(); pos != matched_items.end(); ++pos) {
			storage.modify(pos, renamer);
			++cnt;
			if(!all) break;
		}
		return cnt;
	}

	// update node indexes to match current link content
	auto refresh(const link::id_type& lid) -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  rearrange
	//
	template<Key K>
	auto rearrange(const std::vector<Key_type<K>>& new_order) {
		// convert vector of keys into vector of iterators
		std::vector<std::reference_wrapper<const sp_link>> i_order;
		i_order.reserve(new_order.size());
		for(const auto& k : new_order)
			i_order.push_back(std::ref(*find<K>(k)));
		// apply order
		links_.get<Key_tag<Key::AnyOrder>>().rearrange(i_order.begin());
	}

	///////////////////////////////////////////////////////////////////////////////
	//  misc
	//
	bool accepts(const sp_link& what) const;
	auto accept_object_types(std::vector<std::string> allowed_types) -> void;

	void set_handle(const sp_link& new_handle);

	auto propagate_owner(bool deep) -> void;

	// postprocessing of just inserted link
	// if link points to node, return it
	static sp_node adjust_inserted_link(const sp_link& lnk, const sp_node& target);

	// obtain pointer to owner node
	auto super() const -> sp_node;
	// get node's group ID
	auto gid() const -> std::string;

private:
	node* super_;

	auto erase_impl(
		iterator<Key::ID> key, leaf_postproc_fn ppf = noop_postproc_f,
		bool dont_reset_owner = false
	) -> iterator<Key::ID>;
};
using sp_nimpl = std::shared_ptr<node_impl>;

NAMESPACE_END(blue_sky::tree)
