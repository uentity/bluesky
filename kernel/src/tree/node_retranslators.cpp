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

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

// actor that retranslate some of link's messages attaching a link's ID to them
auto link_retranslator(caf::stateful_actor<rsl_state>* self, caf::group node_grp, lid_type lid)
-> caf::behavior {
	// remember target node group
	self->state.tgt_grp = std::move(node_grp);
	// connect source
	self->state.src_lid = std::move(lid);
	self->state.src_grp = system().groups().get_local(to_string(lid));
	self->join(self->state.src_grp);

	auto sdbg = [=](const std::string& msg_name = {}) {
		auto res = adbg(self, self->state.tgt_grp_id()) << "<- [L] [" << self->state.src_grp_id() << "] ";
		//auto res = caf::aout(self) << self->state.tgt_grp_id() << " <- ";
		if(!msg_name.empty())
			res << '{' << msg_name << "} ";
		return res;
	};
	sdbg() << "retranslator started" << std::endl;

	// register self
	const auto sid = self->id();
	system().registry().put(sid, self);

	// silently drop all other messages not in my character
	self->set_default_handler(caf::drop);

	return {
		// quit after source
		[=](a_bye) {
			self->leave(self->state.src_grp);
			system().registry().erase(sid);
			sdbg() << "retranslator quit" << std::endl;
		},

		// retranslate events
		[=](a_ack, a_lnk_rename, std::string new_name, std::string old_name) {
			sdbg("rename") << old_name << " -> " << new_name << std::endl;
			self->send<high_prio>(
				self->state.tgt_grp, a_ack(), a_lnk_rename(), self->state.src_lid,
				std::move(new_name), std::move(old_name)
			);
		},

		[=](a_ack, a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s) {
			sdbg("status") << to_string(req) << ": " << to_string(prev_s) << " -> " << to_string(new_s);
			self->send<high_prio>(
				self->state.tgt_grp, a_ack(), a_lnk_status(), self->state.src_lid, req, new_s, prev_s
			);
		}
	};
}

// actor that retranslate some of link's messages attaching a link's ID to them
auto node_retranslator(
	caf::stateful_actor<node_rsl_state>* self, caf::group node_grp, lid_type lid, link::actor_type Lactor
) -> caf::behavior {
	// remember target node group
	self->state.tgt_grp = std::move(node_grp);
	// remember source
	self->state.src_lid = lid;
	self->state.src_actor = std::move(Lactor);

	auto sdbg = [=](const std::string& msg_name = {}) {
		auto res = adbg(self, self->state.tgt_grp_id()) << "<- [N] [" << self->state.src_grp_id() << "] ";
		//auto res = caf::aout(self) << self->state.tgt_grp_id() << " <- ";
		if(!msg_name.empty())
			res << '{' << msg_name << "} ";
		return res;
	};
	sdbg() << "retranslator started" << std::endl;

	// register self
	const auto sid = self->id();
	system().registry().put(sid, self);

	// silently drop all other messages not in my character
	self->set_default_handler(caf::drop);

	// auto-start self
	self->send<high_prio>(self, a_apply());

	return {
		// setup retranslator source
		[=](a_apply) {
			// ask for subnode gid from link
			self->request<high_prio>(self->state.src_actor, def_timeout(), a_node_gid())
			.then([=](result_or_errbox<std::string> src_gid) {
				if(src_gid && !src_gid->empty()) {
					// join subnode group
					sdbg() << "going to join node source " << *src_gid << std::endl;
					self->state.src_grp = system().groups().get_local(*src_gid);
					self->join(self->state.src_grp);
					sdbg() << "joined source group " << *src_gid << std::endl;
				}
				else {
					// link doesn't point to node, exit
					sdbg() << "shutdown, src is not a node " << *src_gid << std::endl;
					self->send<high_prio>(self, a_bye());
				}
			});
		},

		// quit following source
		[=](a_bye) {
			self->leave(self->state.src_grp);
			system().registry().erase(sid);
			sdbg() << "retranslator quit" << std::endl;
		},

		// retranslate events
		[=](a_ack, a_lnk_rename, const lid_type& lid, std::string new_name, std::string old_name) {
			sdbg("rename") << old_name << " -> " << new_name << std::endl;
			self->send<high_prio>(
				self->state.tgt_grp, a_ack(), a_lnk_rename(), lid, std::move(new_name), std::move(old_name)
			);
		},

		[=](a_ack, a_lnk_status, const lid_type& lid, Req req, ReqStatus new_s, ReqStatus prev_s) {
			sdbg("status") << to_string(req) << ": " << to_string(prev_s) << " -> " << to_string(new_s);
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_lnk_status(), lid, req, new_s, prev_s);
		},

		[=](a_ack, a_node_insert, const lid_type& lid, std::size_t pos, InsertPolicy pol) {
			sdbg("insert") << to_string(lid) << " in pos " << pos << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_node_insert(), lid, pos, pol);
		},
		[=](a_ack, a_node_insert, const lid_type& lid, std::size_t to_idx, std::size_t from_idx) {
			sdbg("insert move") << to_string(lid) << " pos " << from_idx << " -> " << to_idx << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_node_insert(), lid, to_idx, from_idx);
		},

		[=](a_ack, a_node_erase, lids_v lids, std::vector<std::string> oids) {
			sdbg("erase") << (lids.empty() ? "" : to_string(lids[0])) << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_node_erase(), std::move(lids), oids);
		}
	};
}

NAMESPACE_END(blue_sky::tree)
