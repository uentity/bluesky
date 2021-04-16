/// @file
/// @author uentity
/// @date 31.03.2020
/// @brief Part of link's behavior that process 'ack' messages
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "link_actor.h"
#include <bs/log.h>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)

auto link_actor::make_ack_behavior() -> ack_actor_type::behavior_type {
return {
	// rename ack
	[=](a_ack, a_lnk_rename, std::string new_name, std::string old_name) -> void {
		adbg(this) << "<- [ack] a_lnk_rename: " << old_name << " -> " << new_name << std::endl;
		// retranslate ack to upper level
		ack_up(a_lnk_rename(), std::move(new_name), std::move(old_name));
	},

	// reset status ack
	[=](a_ack, a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s) {
		adbg(this) << "<- [ack] a_lnk_status: " << to_string(req) << " " <<
			to_string(prev_s) << "->" << to_string(new_s) << std::endl;
		// retranslate ack to upper level
		ack_up(a_lnk_status(), req, new_s, prev_s);
	},

	// retranslate pointee node acks to owner's home group
	[=](
		a_ack, caf::actor N, const lid_type& lid,
		a_lnk_rename, std::string new_name, std::string old_name
	) {
		adbg(this) << "<- [ack] [deep] a_lnk_rename" << std::endl;
		forward_up(a_ack(), std::move(N), lid, a_lnk_rename(), std::move(new_name), std::move(old_name));
	},

	[=](
		a_ack, caf::actor N, const lid_type& lid,
		a_lnk_status, Req req, ReqStatus new_rs, ReqStatus old_rs
	) {
		adbg(this) << "<- [ack] [deep] a_lnk_status" << std::endl;
		forward_up(a_ack(), std::move(N), lid, a_lnk_status(), req, new_rs, old_rs);
	},

	[=](a_ack, caf::actor N, const lid_type& lid, a_data, tr_result::box tres) {
		adbg(this) << "<- [ack] [deep] a_data: " << to_string(tres) << std::endl;
		forward_up(a_ack(), std::move(N), lid, a_data(), std::move(tres));
	},

	[=](a_ack, caf::actor N, a_node_insert, const lid_type& lid, size_t pos) {
		adbg(this) << "<- [ack] [deep] a_node_insert" << std::endl;
		forward_up(a_ack(), std::move(N), a_node_insert(), lid, pos);
	},

	[=](a_ack, caf::actor N, a_node_insert, const lid_type& lid, size_t pos1, size_t pos2) {
		adbg(this) << "<- [ack] [deep] a_node_insert [move]" << std::endl;
		forward_up(a_ack(), std::move(N), a_node_insert(), lid, pos1, pos2);
	},

	[=](a_ack, caf::actor N, a_node_erase, lids_v erased_leafs) {
		adbg(this) << "<- [ack] [deep] a_node_erase" << std::endl;
		forward_up(a_ack(), std::move(N), a_node_erase(), std::move(erased_leafs));
	}
}; }

NAMESPACE_END(blue_sky::tree)
