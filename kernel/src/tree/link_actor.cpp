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

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
using namespace std::chrono_literals;

// global random UUID generator for BS links
static boost::uuids::random_generator gen;
/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
link_actor::link_actor(caf::actor_config& cfg, sp_limpl Limpl)
	: super(cfg), pimpl_(std::move(Limpl)), impl([this]() -> link_impl& {
		if(!pimpl_) throw error{"link actor: bad (null) link impl passed"};
		return *pimpl_;
	}())
{
	// remember link's local group
	if(cfg.groups->begin() != cfg.groups->end())
		impl.self_grp = *cfg.groups->begin();
	else {
		auto grp_id = to_string(impl.id_);
		impl.self_grp = system().groups().get_local(grp_id);
		join(impl.self_grp);
	}
	pdbg() << "link joined self group " << impl.self_grp.get()->identifier() << std::endl;

	// on exit say goodbye to self group
	set_exit_handler([this](caf::exit_msg& er) {
		goodbye();
		default_exit_handler(this, er);
	});

	//bind_new_id();
}

link_actor::~link_actor() = default;

auto link_actor::goodbye() -> void {
	if(impl.self_grp) {
		// say goodbye to self group
		send(impl.self_grp, a_bye());
		leave(impl.self_grp);
		aout(this) << "link left self group " << impl.self_grp.get()->identifier() << std::endl;
		//	<< "\n" << kernel::tools::get_backtrace(30, 4) << std::endl;
	}
}

auto link_actor::bind_new_id() -> void {
	// leave old group
	auto new_id = to_string(impl.id_);
	auto dout = aout(this);
	dout << "link bind: ";
	if(impl.self_grp) {
		if(new_id == impl.self_grp.get()->identifier()) return;
		// rebind friends to new ID
		// [NOTE] don't send bye, otherwise retranslators from this will quit
		send(impl.self_grp, a_bind_id(), impl.id_);
		leave(impl.self_grp);
		dout << impl.self_grp.get()->identifier();
	}
	else {
		// create self local group & join into it
		impl.self_grp = system().groups().get_local(new_id);
		join(impl.self_grp);
	}
	// say hello to everybody
	send(impl.self_grp, a_hi());
	dout << "-> " << new_id << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
//  manual status management
//
auto link_actor::pdbg() -> caf::actor_ostream {
	auto guard = std::shared_lock{impl.guard_};
	return caf::aout(this) << to_string(impl.id_) << " "; // << master_->type_id() << ": ";
}

auto link_actor::rs_reset(
	Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs
) -> ReqStatus {
	return impl.rs_reset(
		request, cond, new_rs, old_rs,
		[this](Req req, ReqStatus new_s, ReqStatus old_s) {
			send(impl.self_grp, a_lnk_status(), a_ack(), req, new_s, old_s);
		}
	);
}

///////////////////////////////////////////////////////////////////////////////
//  data & data_node
//
auto link_actor::data_ex(bool wait_if_busy) -> result_or_err<sp_obj> {
	//pdbg() << "aimpl: member data_ex() status = " <<
	//	(int)impl.status_[0].value << (int)impl.status_[1].value << std::endl;

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
	//pdbg() << "aimpl: member data_node_ex() status = " <<
	//	(int)impl.status_[0].value << (int)impl.status_[1].value << std::endl;

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
	[](a_bye) { },

	/// get id
	[this](a_lnk_id) {
		// ID change is very special op (only by deserialization), so don't lock
		return impl.id_;
	},

	/// update self ID from sibling link in self group
	[](a_bind_id, const link::id_type&) {
		// [TODO] add impl
	},

	/// get name
	[this](a_lnk_name) {
		auto guard = std::shared_lock{impl.guard_};
		return impl.name_;
	},

	/// rename
	[this](a_lnk_rename, std::string new_name, bool silent) {
		auto solo = std::unique_lock{ impl.guard_ };
		auto old_name = impl.name_;
		impl.name_ = std::move(new_name);
		// send rename ack message
		if(!silent) {
			//pdbg() << "=> lnk_rename: = " << impl.name_ << " -> " << new_name <<
			//	(silent ? " silent" : " loud") << std::endl;
			send(impl.self_grp, a_lnk_rename(), a_ack(), impl.name_, std::move(old_name));
		}
	},
	// rename ack
	[this](a_lnk_rename, a_ack, std::string new_name, const std::string& old_name) {
		//pdbg() << "=> lnk_rename ack: = " << old_name << " -> " << new_name << std::endl;
		if(current_sender() != this) {
			send(this, a_lnk_rename(), std::move(new_name), true);
		}
	},

	// change status
	[this](a_lnk_status, Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs) {
		return rs_reset(req, cond, new_rs, prev_rs);
	},

	// [NOTE] nop for a while
	[](a_lnk_status, a_ack, Req req, ReqStatus new_s, ReqStatus prev_s) {
		//pdbg() << " => a_lnk_status ack: " <<
		//	(req == Req::Data ? " Data " : " DataNode ") <<
		//	int(prev_s) << " -> " << int(new_s) << std::endl;
	},

	// get oid
	[this](a_lnk_oid) -> std::string {
		//auto obj = data_ex(false);
		auto res = data_ex(false)
		.map([](const sp_obj& obj) {
			return obj->id();
		}).value_or(nil_oid);
		pdbg() << "=> a_lnk_oid: " << res << std::endl;
		//pdbg() << "=> a_lnk_oid: " << (obj ? (void*)obj.value().get() : (void*)0) << " " << res << std::endl;
		return res;
	},

	// get object type_id
	[this](a_lnk_otid) -> std::string {
		//pdbg() << "aimpl: obj type id()" << std::endl;
		return data_ex(false)
		.map([](const sp_obj& obj) {
			return obj->type_id();
		}).value_or(nil_oid);
	},

	// obtain inode
	[this](a_lnk_inode) -> result_or_errbox<inodeptr> {
		//pdbg() << "aimpl: inode()" << std::endl;
		return impl.get_inode();
	},

	// default handler for `data_node` that works via `data`
	[this](a_lnk_dnode, bool wait_if_busy) -> result_or_errbox<sp_node> {
		//pdbg() << "aimpl: data_node() status = " <<
		//	(int)impl.status_[0].value << (int)impl.status_[1].value << std::endl;

		return data_node_ex(wait_if_busy);
	},

	// default `data` handler calls virtual `data()` function that by default returns nullptr
	[this](a_lnk_data, bool wait_if_busy) -> caf::result< result_or_errbox<sp_obj> > {
		//pdbg() << "aimpl: data() status =" <<
		//	(int)impl.status_[0].value << (int)impl.status_[1].value << std::endl;

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
