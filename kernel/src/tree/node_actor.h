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
#include <bs/kernel/radio.h>
#include <bs/detail/enumops.h>
#include <bs/detail/sharded_mutex.h>

#include "actor_common.h"
#include "node_impl.h"

#include <set>
#include <unordered_map>
#include <optional>

#include <boost/uuid/uuid_hash.hpp>

// iterators are only passed by `node` API and must not be serialized
#define OMIT_ITERATORS_SERIALIZATION_(K)                         \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::node::iterator<K>) \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::node::range<K>)

#define OMIT_ITERATORS_SERIALIZATION                               \
OMIT_ITERATORS_SERIALIZATION_(blue_sky::tree::node::Key::ID)       \
OMIT_ITERATORS_SERIALIZATION_(blue_sky::tree::node::Key::OID)      \
OMIT_ITERATORS_SERIALIZATION_(blue_sky::tree::node::Key::Name)     \
OMIT_ITERATORS_SERIALIZATION_(blue_sky::tree::node::Key::Type)     \
OMIT_ITERATORS_SERIALIZATION_(blue_sky::tree::node::Key::AnyOrder) \

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

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
class BS_HIDDEN_API node_actor : public caf::event_based_actor, bs_detail::sharded_mutex<std::mutex> {
public:
	friend struct access_node_actor;
	using super = caf::event_based_actor;

	// holds reference to node impl
	sp_nimpl pimpl_;
	node_impl& impl;
	// map links to retranslator actors
	using axon_t = std::pair<std::uint64_t, std::optional<std::uint64_t>>;
	std::unordered_map<link::id_type, axon_t> axons_;

	node_actor(caf::actor_config& cfg, caf::group ngrp, sp_nimpl Nimpl);
	virtual ~node_actor();

	// say goodbye to others & leave self group
	auto goodbye() -> void;

	auto name() const -> const char* override;
	auto make_behavior() -> behavior_type override;

	auto insert(
		sp_link L, const InsertPolicy pol, bool silent = false
	) -> insert_status<Key::ID>;

	auto insert(
		sp_link L, std::size_t idx, const InsertPolicy pol, bool silent = false
	) -> std::pair<size_t, bool>;

	enum EraseOpts { Normal = 0, Silent = 1, DontResetOwner = 2 };
	auto erase(const link::id_type& key, EraseOpts opts = EraseOpts::Normal) -> size_t;
	auto erase(std::size_t idx) -> size_t;
	auto erase(const std::string& key, Key key_meaning = Key::Name) -> size_t;

	auto retranslate_from(const sp_link& L) -> void;
	auto stop_retranslate_from(const sp_link& L) -> void;
	// stops retranslating from all leafs
	auto disconnect() -> void;
};

// helper for correct spawn of node actor
template<typename Actor = node_actor, caf::spawn_options Os = caf::no_spawn_options, class... Ts>
inline auto spawn_nactor(std::shared_ptr<node_impl> nimpl, const std::string& gid, Ts&&... args) {
	auto& AS = kernel::radio::system();
	// create unique UUID for node's group because object IDs can be non-unique
	auto ngrp = AS.groups().get_local(gid);
	return AS.spawn_in_group<Actor, Os>(
		ngrp, ngrp, std::move(nimpl), std::forward<Ts>(args)...
	);
}

NAMESPACE_END(blue_sky::tree)

BS_ALLOW_ENUMOPS(blue_sky::tree::node_actor::EraseOpts)
