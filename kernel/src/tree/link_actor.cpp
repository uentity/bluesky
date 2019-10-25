/// @file
/// @author uentity
/// @date 09.07.2019
/// @brief Base link actor implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "link_actor.h"
#include <bs/kernel/tools.h>

#include <bs/kernel/config.h>
#include <bs/kernel/radio.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <boost/uuid/uuid_generators.hpp>

#define DEBUG_ACTOR 1

#if DEBUG_ACTOR == 1
#include <caf/actor_ostream.hpp>
#else
#include <fstream>
#endif

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
using namespace std::chrono_literals;

NAMESPACE_BEGIN()
#if DEBUG_ACTOR == 1

auto adbg(link_actor* A) -> caf::actor_ostream {
	return caf::aout(A) << "link " << to_string(A->impl.id_) <<
		", name '" << A->impl.name_ << "': ";
}

#else

auto adbg(link_actor*) -> std::ostream& {
	static auto ignore = std::ofstream{"/dev/null"};
	return ignore;
}

#endif
NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
link_actor::link_actor(caf::actor_config& cfg, caf::group lgrp, sp_limpl Limpl)
	: super(cfg), pimpl_(std::move(Limpl)), impl([this]() -> link_impl& {
		if(!pimpl_) throw error{"link actor: bad (null) link impl passed"};
		return *pimpl_;
	}())
{
	// remember link's local group
	impl.self_grp = std::move(lgrp);
	adbg(this) << "joined self group " << impl.self_grp.get()->identifier() << std::endl;

	// on exit say goodbye to self group
	set_exit_handler([this](caf::exit_msg& er) {
		goodbye();
		default_exit_handler(this, er);
	});

	// prevent termination in case some errors happens in group members
	// for ex. if they receive unexpected messages (translators normally do)
	set_error_handler([this](caf::error er) {
		switch(static_cast<caf::sec>(er.code())) {
		case caf::sec::unexpected_message :
			break;
		default:
			default_error_handler(this, er);
		}
	});

	set_default_handler(caf::drop);
}

link_actor::~link_actor() = default;

auto link_actor::name() const -> const char* {
	return "link_actor";
}

auto link_actor::on_exit() -> void {
	adbg(this) << "dies" << std::endl;
}

auto link_actor::goodbye() -> void {
	if(impl.self_grp) {
		// say goodbye to self group
		send(impl.self_grp, a_bye());
		leave(impl.self_grp);
		adbg(this) << "left self group " << impl.self_grp.get()->identifier() << std::endl;
		//	<< "\n" << kernel::tools::get_backtrace(30, 4) << std::endl;
	}
}

///////////////////////////////////////////////////////////////////////////////
//  data & data_node
//
auto link_actor::data_ex(bool wait_if_busy) -> result_or_err<sp_obj> {
	// never returns NULL object
	return link_invoke(
		pimpl_.get(),
		[](link_impl* limpl) { return limpl->data(); },
		impl.status_[0], wait_if_busy,
		// send status changed message
		function_view{ [this](ReqStatus new_v, ReqStatus prev_v) {
			auto guard = std::shared_lock{impl.guard_};
			if(prev_v != new_v) send(impl.self_grp, a_lnk_status(), a_ack(), Req::Data, new_v, prev_v);
		} }
	).and_then([](sp_obj&& obj) {
		return obj ?
			result_or_err<sp_obj>(std::move(obj)) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

auto link_actor::data_node_ex(bool wait_if_busy) -> result_or_err<sp_node> {
	// never returns NULL node
	return link_invoke(
		this,
		[](link_actor* lnk) { return lnk->data_node(); },
		impl.status_[1], wait_if_busy,
		// send status changed message
		function_view{ [this](ReqStatus new_v, ReqStatus prev_v) {
			auto guard = std::shared_lock{impl.guard_};
			if(prev_v != new_v) send(impl.self_grp, a_lnk_status(), a_ack(), Req::DataNode, new_v, prev_v);
		} }
	).and_then([](sp_node&& N) {
		return N ?
			result_or_err<sp_node>(std::move(N)) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

auto link_actor::data_node() -> result_or_err<sp_node> {
	return data_ex().and_then([](sp_obj&& obj) {
		// don't check if obj is nullptr, because data_ex() never returns NULL
		return obj->is_node() ?
			result_or_err<sp_node>(std::static_pointer_cast<tree::node>(std::move(obj))) :
			tl::make_unexpected(error::quiet(Error::NotANode));
	});
}

///////////////////////////////////////////////////////////////////////////////
//  behavior
//
auto link_actor::make_behavior() -> behavior_type {
	return make_generic_behavior();
}

auto link_actor::make_generic_behavior() -> behavior_type { return {
	/// skip `bye` message (should always come from myself)
	[this](a_bye) {
		adbg(this) << "<- a_lnk_bye" << std::endl;
	},

	/// get id
	[this](a_lnk_id) {
		// ID change is very special op (only by deserialization), so don't lock
		adbg(this) << "<- a_lnk_id: " << to_string(impl.id_) << std::endl;
		return impl.id_;
	},

	/// get name
	[this](a_lnk_name) {
		auto guard = std::shared_lock{impl.guard_};
		adbg(this) << "<- a_lnk_name: " << impl.name_ << std::endl;
		return impl.name_;
	},

	/// rename
	[this](a_lnk_rename, std::string new_name, bool silent) {
		auto solo = std::unique_lock{ impl.guard_ };
		adbg(this) << "<- a_lnk_rename " << (silent ? "silent: " : "loud: ") << impl.name_ <<
			" -> " << new_name << std::endl;

		auto old_name = impl.name_;
		impl.name_ = std::move(new_name);
		// send rename ack message
		if(!silent) {
			send(impl.self_grp, a_lnk_rename(), a_ack(), impl.name_, std::move(old_name));
		}
	},
	// rename ack
	[this](a_lnk_rename, a_ack, std::string new_name, const std::string& old_name) {
		adbg(this) << "<- a_lnk_rename ack: " << old_name << " -> " << new_name << std::endl;
		if(current_sender() != this) {
			send(this, a_lnk_rename(), std::move(new_name), true);
		}
	},

	// change status
	[this](a_lnk_status, Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs) {
		adbg(this) << "<- a_lnk_status: " <<
			(req == Req::Data ? " Data " : " DataNode ") <<
			int(prev_rs) << " -> " << int(new_rs) << std::endl;
		return impl.rs_reset(
			req, cond, new_rs, prev_rs,
			[this](Req req, ReqStatus new_s, ReqStatus old_s) {
				send(impl.self_grp, a_lnk_status(), a_ack(), req, new_s, old_s);
			}
		);
	},

	// [NOTE] nop for a while
	[this](a_lnk_status, a_ack, Req req, ReqStatus new_s, ReqStatus prev_s) {
		adbg(this) << "<- a_lnk_status ack: " <<
			(req == Req::Data ? " Data " : " DataNode ") <<
			int(prev_s) << " -> " << int(new_s) << std::endl;
	},

	// get oid
	[this](a_lnk_oid) -> std::string {
		//auto obj = data_ex(false);
		auto res = data_ex(false)
		.map([](const sp_obj& obj) {
			return obj->id();
		}).value_or(nil_oid);
		adbg(this) << "<- a_lnk_oid: " << res << std::endl;
		//adbg(this) << "=> a_lnk_oid: " << (obj ? (void*)obj.value().get() : (void*)0) << " " << res << std::endl;
		return res;
	},

	// get object type_id
	[this](a_lnk_otid) -> std::string {
		//return data_ex(false)
		auto res = data_ex(false)
		.map([](const sp_obj& obj) {
			return obj->type_id();
		}).value_or(type_descriptor::nil().name);
		adbg(this) << "<- a_lnk_otid: " << res << std::endl;
		return res;
	},

	// obtain inode
	[this](a_lnk_inode) -> result_or_errbox<inodeptr> {
		adbg(this) << "<- a_lnk_inode" << std::endl;
		return impl.get_inode();
	},

	// default handler for `data_node` that works via `data`
	[this](a_lnk_dnode, bool wait_if_busy) -> result_or_errbox<sp_node> {
		adbg(this) << "<- a_lnk_dnode, status = " <<
			(int)impl.status_[0].value << (int)impl.status_[1].value << std::endl;

		return data_node_ex(wait_if_busy);
	},

	// default `data` handler calls virtual `data()` function that by default returns nullptr
	[this](a_lnk_data, bool wait_if_busy) -> result_or_errbox<sp_obj> {
		adbg(this) << "<- a_lnk_data, status = " <<
			(int)impl.status_[0].value << (int)impl.status_[1].value << std::endl;

		return data_ex(wait_if_busy);
	}
}; }

///////////////////////////////////////////////////////////////////////////////
//  simple_link_actor
//
auto simple_link_actor::data_ex(bool wait_if_busy) -> result_or_err<sp_obj> {
	return impl.data()
	.and_then([](sp_obj&& obj) {
		return obj ?
			result_or_err<sp_obj>(std::move(obj)) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

auto simple_link_actor::data_node_ex(bool wait_if_busy) -> result_or_err<sp_node> {
	return data_node();
}

/*-----------------------------------------------------------------------------
 *  other
 *-----------------------------------------------------------------------------*/
// extract timeout from kernel config
auto def_timeout(bool for_data) -> caf::duration {
	using namespace kernel::config;
	return caf::duration{ for_data ?
		get_or( config(), "radio.data-timeout", def_data_timeout ) :
		get_or( config(), "radio.timeout", timespan{100ms} )
	};
}

NAMESPACE_END(blue_sky::tree)
