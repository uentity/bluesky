/// @file
/// @author uentity
/// @date 31.03.2020
/// @brief Part of node's behavior that process 'ack' messages
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_actor.h"

#include <bs/log.h>
#include <bs/tree/tree.h>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)

auto node_actor::make_ack_behavior() -> ack_actor_type::behavior_type {
return {
	// ack on insert
	[=](a_ack, caf::actor origin, a_node_insert, const lid_type& lid, size_t pos) {
		adbg(this) << "{a_node_insert ack}: " << pos << std::endl;
		// notify handle's home about data change (to just trigger event handlers)
		if(origin == this)
			forward_up_home(a_ack(), a_data(), tr_result::box{});
		forward_up(a_ack(), std::move(origin), a_node_insert(), lid, pos);

		//if(origin != this) {
		//	request(origin, impl.timeout, a_node_find(), lid)
		//	.then([=](link L) {
		//		// [NOTE] silent insert
		//		insert(std::move(L), pos, pol, true);
		//	});
		//}
	},
	// ack on move
	[=](a_ack, caf::actor origin, a_node_insert, const lid_type& lid, size_t to, size_t from) {
		adbg(this) << "{a_node_insert ack} [move]: " << from << " -> " << to << std::endl;
		// notify handle's home about data change (to just trigger event handlers)
		if(origin == this)
			forward_up_home(a_ack(), a_data(), tr_result::box{});
		forward_up(a_ack(), std::move(origin), a_node_insert(), lid, to, from);

		//if(origin != this) {
		//	if(auto p = impl.find<Key::ID, Key::ID>(lid); p != impl.end<Key::ID>()) {
		//		insert(*p, to, InsertPolicy::AllowDupNames, true);
		//	}
		//}
	},

	// ack on erase
	[=](a_ack, caf::actor origin, a_node_erase, lids_v lids) {
		adbg(this) << "{a_node_erase ack}" << std::endl;
		// notify handle about data change
		if(origin == this)
			forward_up_home(a_ack(), a_data(), tr_result::box{});
		forward_up(a_ack(), std::move(origin), a_node_erase(), std::move(lids));

		//if(auto S = current_sender(); S != this && !lids.empty()) {
		//	erase(lids.front(), EraseOpts::Silent);
		//}
	},

	// handle my leaf rename
	[=](a_ack, const lid_type& lid, a_lnk_rename, std::string new_, std::string old_) {
		adbg(this) << "{a_lnk_rename ack}" << std::endl;
		ack_up(lid, a_lnk_rename(), std::move(new_), std::move(old_));
		// notify handle about data change
		forward_up_home(a_ack(), a_data(), tr_result::box{});
	},
	// track my leaf status
	[=](a_ack, const lid_type& lid, a_lnk_status, Req req, ReqStatus new_, ReqStatus old_) {
		ack_up(lid, a_lnk_status(), req, new_, old_);
	},
	// my leafs data change
	[=](a_ack, const lid_type& lid, a_data, tr_result::box tres) {
		ack_up(lid, a_data(), std::move(tres));
	},

	// retranslate deep links acks
	[=](
		a_ack, caf::actor N, const lid_type& lid,
		a_lnk_rename, std::string new_name, std::string old_name
	) {
		forward_up(a_ack(), std::move(N), lid, a_lnk_rename(), std::move(new_name), std::move(old_name));
	},

	[=](
		a_ack, caf::actor N, const lid_type& lid,
		a_lnk_status, Req req, ReqStatus new_rs, ReqStatus old_rs
	) {
		forward_up(a_ack(), std::move(N), lid, a_lnk_status(), req, new_rs, old_rs);
	},

	[=](a_ack, caf::actor N, const lid_type& lid, a_data, tr_result::box tres) {
		forward_up(a_ack(), std::move(N), lid, a_data(), std::move(tres));
	}
}; }

NAMESPACE_END(blue_sky::tree)
