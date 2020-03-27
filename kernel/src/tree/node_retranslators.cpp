/// @file
/// @author uentity
/// @date 30.01.2020
/// @brief Node messages retranslators impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_retranslators.h"
#include "link_impl.h"

#include <bs/kernel/radio.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

#include <boost/uuid/uuid_io.hpp>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

#if DEBUG_ACTOR == 1

template<typename State>
static auto adbg(caf::stateful_actor<State>* A, const std::string& msg_name = {}) {
	auto& S = A->state;
	auto res = caf::aout(A);
	if(auto nid = S.tgt_grp_id(); !nid.empty())
		res << "[N] [" << nid << "]: ";
	if(auto sid = S.src_grp_id(); !sid.empty()) {
		if constexpr(std::is_same_v<State, node_rsl_state>)
			res << "<- [N]";
		else
			res << "<- [L]";
		res << " [" << sid << "] ";
	}
	if(!msg_name.empty())
		res << '{' << msg_name << "} ";
	return res;
}

#endif

// actor that retranslate some of link's messages attaching a link's ID to them
auto node_retranslator(
	caf::stateful_actor<node_rsl_state>* self, caf::group node_grp, lid_type lid, link::actor_type Lactor
) -> caf::behavior {
	// remember target node group
	self->state.tgt_grp = std::move(node_grp);
	// remember source
	self->state.src_lid = lid;
	self->state.src_actor = std::move(Lactor);

	// register self
	const auto sid = self->id();
	system().registry().put(sid, self);

	// silently drop all other messages not in my character
	self->set_default_handler(caf::drop);

	// auto-start self
	self->send<high_prio>(self, a_apply());

	adbg(self) << "retranslator started" << std::endl;
	return {
		// setup retranslator source
		[=](a_apply) {
			// ask for subnode gid from link
			self->request<high_prio>(self->state.src_actor, def_timeout(), a_node_gid())
			.then([=](result_or_errbox<std::string> src_gid) {
				if(src_gid && !src_gid->empty()) {
					// join subnode group
					adbg(self) << "going to join node source " << *src_gid << std::endl;
					self->state.src_grp = system().groups().get_local(*src_gid);
					self->join(self->state.src_grp);
					adbg(self) << "joined source group " << *src_gid << std::endl;
				}
				else {
					// link doesn't point to node, exit
					adbg(self) << "shutdown, src is not a node " << *src_gid << std::endl;
					self->send<high_prio>(self, a_bye());
				}
			});
		},

		// quit following source
		[=](a_bye) {
			self->leave(self->state.src_grp);
			system().registry().erase(sid);
			adbg(self) << "retranslator quit" << std::endl;
		},

		// retranslate events
		[=](a_ack, a_lnk_rename, const lid_type& lid, std::string new_name, std::string old_name) {
			adbg(self, "rename") << old_name << " -> " << new_name << std::endl;
			self->send<high_prio>(
				self->state.tgt_grp, a_ack(), a_lnk_rename(), lid, std::move(new_name), std::move(old_name)
			);
		},

		[=](a_ack, a_lnk_status, const lid_type& lid, Req req, ReqStatus new_s, ReqStatus prev_s) {
			adbg(self, "status")
				<< to_string(req) << ": " << to_string(prev_s) << " -> " << to_string(new_s) << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_lnk_status(), lid, req, new_s, prev_s);
		},

		[=](a_ack, a_node_insert, const lid_type& lid, std::size_t pos, InsertPolicy pol) {
			adbg(self, "insert") << to_string(lid) << " in pos " << pos << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_node_insert(), lid, pos, pol);
		},
		[=](a_ack, a_node_insert, const lid_type& lid, std::size_t to_idx, std::size_t from_idx) {
			adbg(self, "insert move") << to_string(lid) << " pos " << from_idx << " -> " << to_idx << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_node_insert(), lid, to_idx, from_idx);
		},

		[=](a_ack, a_node_erase, lids_v lids, std::vector<std::string> oids) {
			adbg(self, "erase") << (lids.empty() ? "" : to_string(lids[0])) << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_node_erase(), std::move(lids), oids);
		}
	};
}

NAMESPACE_END(blue_sky::tree)
