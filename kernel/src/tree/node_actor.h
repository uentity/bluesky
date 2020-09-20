/// @file
/// @author uentity
/// @date 29.06.2018
/// @brief BS tree node implementation part of PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/atoms.h>
#include <bs/actor_common.h>
#include <bs/tree/node.h>
#include <bs/kernel/radio.h>

#include "node_impl.h"

#include <boost/uuid/uuid_hash.hpp>

#include <caf/actor_cast.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

/*-----------------------------------------------------------------------------
 *  node_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API node_actor : public caf::event_based_actor {
public:
	using super = caf::event_based_actor;
	using actor_type = node_impl::actor_type;
	using primary_actor_type = node_impl::primary_actor_type;
	using ack_actor_type = node_impl::ack_actor_type;

	// holds reference to node impl
	sp_nimpl pimpl_;
	node_impl& impl;

	node_actor(caf::actor_config& cfg, caf::group nhome, sp_nimpl Nimpl);
	~node_actor();

	// get typed node actor handle
	inline auto actor() -> node_impl::actor_type {
		return caf::actor_cast<node_impl::actor_type>(this);
	}

	// pass message to upper (owner) level of tree structure
	template<typename... Args>
	auto forward_up(Args&&... args) -> void {
		// node forward messages to handle's actor
		if(auto h = impl.handle())
			send(link_impl::actor(h), std::forward<Args>(args)...);
	}

	// pass message exactly to home group of owning handle link
	template<typename... Args>
	auto forward_up_home(Args&&... args) -> void {
		// node forward messages to handle's actor
		if(auto h = impl.handle())
			checked_send<link_impl::home_actor_type>(*this, h.home(), std::forward<Args>(args)...);
	}

	// forward 'ack' message to upper level, auto prepend it with this node actor handle
	template<typename... Args>
	auto ack_up(Args&&... args) -> void {
		forward_up(a_ack(), this, std::forward<Args>(args)...);
	}

	auto on_exit() -> void override;

	auto name() const -> const char* override;

	// say goodbye to others & leave home
	auto goodbye() -> void;

	// parts of behavior
	auto make_primary_behavior() -> primary_actor_type::behavior_type;
	auto make_ack_behavior() -> ack_actor_type::behavior_type;
	// combines primary + ack
	auto make_behavior() -> behavior_type override;

	///////////////////////////////////////////////////////////////////////////////
	//  leafs operations that require actor
	//
	auto rename(std::vector<iterator<Key::Name>> namesakes, const std::string& new_name)
	-> caf::result<std::size_t>;

	template<Key K = Key::ID>
	auto rename(const Key_type<K>& key, const std::string& new_name) -> caf::result<std::size_t> {
		return rename(impl.equal_range<K>(key).template extract_it<iterator<Key::Name>>([&](auto& p) {
			return impl.project<K, Key::Name>(p);
		}), new_name);
	}

	auto insert(link L, InsertPolicy pol) -> caf::response_promise;

	auto insert(link L, std::size_t idx, InsertPolicy pol) -> caf::response_promise;

	auto insert(links_v Ls, InsertPolicy pol) -> caf::result<std::size_t>;

	auto erase(const lid_type& key, EraseOpts opts = EraseOpts::Normal) -> size_t;
};

// helper for correct spawn of node actor
template<typename Actor = node_actor, caf::spawn_options Os = caf::no_spawn_options, class... Ts>
inline auto spawn_nactor(std::shared_ptr<node_impl> nimpl, caf::group nhome, Ts&&... args) {
	auto& AS = kernel::radio::system();
	return AS.spawn_in_group<Actor, Os>(nhome, nhome, std::move(nimpl), std::forward<Ts>(args)...);
}

NAMESPACE_END(blue_sky::tree)
