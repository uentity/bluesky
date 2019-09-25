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

#include <bs/kernel/radio.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <boost/uuid/uuid_generators.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

// global random UUID generator for BS links
static boost::uuids::random_generator gen;
/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
link_actor::link_actor(caf::actor_config& cfg, std::string name, Flags f, timespan data_timeout)
	: super(cfg), id_(gen()), name_(std::move(name)), flags_(f), timeout_(def_data_timeout)
{
	bind_new_id();
}

link_actor::~link_actor() = default;

auto link_actor::goodbye() -> void {
	if(self_grp) {
		// say goodbye to self group
		send(self_grp, a_bye());
		leave(self_grp);
		//aout(this) << "link left self group " << self_grp.get()->identifier() << std::endl;
		//	<< "\n" << kernel::tools::get_backtrace(30, 4) << std::endl;
	}
}

auto link_actor::bind_new_id() -> void {
	// inform friends about ID change
	// [NOTE] don't send bye, otherwise retranslators from this will quit
	send(self_grp, a_bind_id(), id_);
	// create self local group & join into it
	self_grp = system().groups().get_local( to_string(id_) );
	join(self_grp);
	//aout(this) << "link joined self group " << self_grp.get()->identifier() << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
//  manual status management
//
auto link_actor::pdbg() -> caf::actor_ostream {
	return caf::aout(this) << to_string(id_); //<< " " << master_->type_id() << ": ";
}

auto link_actor::req_status(Req request) const -> ReqStatus {
	if(const auto i = (unsigned)request; i < 2)
		return status_[i].value;
	return ReqStatus::Void;
}

auto link_actor::rs_reset(
	Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs
) -> ReqStatus {
	const auto i = (unsigned)request;
	if(i >= 2) return ReqStatus::Error;

	// atomic set value
	auto& S = status_[i];
	auto guard = scope_atomic_flag(S.flag);
	const auto self = S.value;
	if( cond == ReqReset::Always ||
		(cond == ReqReset::IfEq && self == old_rs) ||
		(cond == ReqReset::IfNeq && self != old_rs)
	) {
		S.value = new_rs;
		// Data = OK will always fire (work as 'data changed' signal)
		if(
			!bool(cond & ReqReset::Silent) &&
			(new_rs != self || (request == Req::Data && new_rs == ReqStatus::OK))
		)
			send(self_grp, a_lnk_status(), a_ack(), request, new_rs, self);
	}
	return self;
}

auto link_actor::reset_owner(const sp_node& new_owner) -> void {
	auto guard = std::lock_guard{solo_};
	owner_ = new_owner;
}

///////////////////////////////////////////////////////////////////////////////
//  inode
//
auto link_actor::get_inode() -> result_or_err<inodeptr> {
	// default implementation obtains inode from `data_ex()->inode_`
	return data_ex().and_then([](const sp_obj& obj) {
		return obj ?
			result_or_err<inodeptr>(obj->inode_.lock()) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

auto link_actor::make_inode(const sp_obj& obj, inodeptr new_i) -> inodeptr {
	if(!obj) return nullptr;

	auto obj_i = obj->inode_.lock();
	if(!obj_i) {
		obj_i = new_i ? std::move(new_i) : std::make_shared<inode>();
		obj->inode_ = obj_i;
	}
	else if(new_i) {
		*obj_i = *new_i;
	}
	return obj_i;
}

///////////////////////////////////////////////////////////////////////////////
//  data & data_node
//
auto link_actor::data_ex(bool wait_if_busy) -> result_or_err<sp_obj> {
	//pdbg() << "aimpl: member data_ex() status = " <<
	//	(int)status_[0].value << (int)status_[1].value << std::endl;

	// never returns NULL object
	return link_invoke(
		this,
		[](link_actor* lnk) { return lnk->data(); },
		status_[0], wait_if_busy,
		// send status changed message
		function_view{ [this](ReqStatus new_v, ReqStatus prev_v) {
			if(prev_v != new_v) send(self_grp, a_lnk_status(), a_ack(), Req::Data, new_v, prev_v);
		} }
	).and_then([](sp_obj&& obj) {
		return obj ?
			result_or_err<sp_obj>(std::move(obj)) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

auto link_actor::data_node_ex(bool wait_if_busy) -> result_or_err<sp_node> {
	//pdbg() << "aimpl: member data_node_ex() status = " <<
	//	(int)status_[0].value << (int)status_[1].value << std::endl;

	// never returns NULL node
	return link_invoke(
		this,
		[](link_actor* lnk) { return lnk->data_node(); },
		status_[1], wait_if_busy,
		// send status changed message
		function_view{ [this](ReqStatus new_v, ReqStatus prev_v) {
			if(prev_v != new_v) send(self_grp, a_lnk_status(), a_ack(), Req::DataNode, new_v, prev_v);
		} }
	).and_then([](sp_node&& N) {
		return N ?
			result_or_err<sp_node>(std::move(N)) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

auto link_actor::data_node() -> result_or_err<sp_node> {
	return data_ex().and_then([](const sp_obj& obj) {
		// don't check if obj is nullptr, because data_ex() never returns NULL
		return obj->is_node() ?
			result_or_err<sp_node>(std::static_pointer_cast<tree::node>(obj)) :
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
	/// skip `id bind` and `bye` message (should always come from myself)
	[](a_bind_id, const link::id_type&) {},
	[](a_bye) {},

	/// rename
	[this](a_lnk_rename, std::string new_name, bool silent) {
		auto solo = std::lock_guard{ solo_ };
		auto old_name = name_;
		name_ = std::move(new_name);
		// send rename ack message
		if(!silent) {
			//pdbg() << "=> lnk_rename: = " << name_ << " -> " << new_name <<
			//	(silent ? " silent" : " loud") << std::endl;
			send(self_grp, a_lnk_rename(), a_ack(), name_, std::move(old_name));
		}
	},
	// rename ack
	[this](a_lnk_rename, a_ack, std::string new_name, const std::string& old_name) {
		//pdbg() << "=> lnk_rename ack: = " << old_name << " -> " << new_name << std::endl;
		if(current_sender() != this) {
			send(this, a_lnk_rename(), std::move(new_name), true);
		}
	},

	// [NOTE] nop for a while
	[](a_lnk_status, a_ack, Req req, ReqStatus new_s, ReqStatus prev_s) {
		//pdbg() << " => a_lnk_status ack: " <<
		//	(req == Req::Data ? " Data " : " DataNode ") <<
		//	int(prev_s) << " -> " << int(new_s) << std::endl;
	},

	// get oid
	[this](a_lnk_oid) -> std::string {
		//pdbg() << "=> a_lnk_oid" << std::endl;

		if(req_status(Req::Data) == ReqStatus::OK) {
			if(auto D = data(); D && *D) return (*D)->id();
		}
		return to_string(boost::uuids::nil_uuid());
	},

	// get object type_id
	[this](a_lnk_otid) -> std::string {
		//pdbg() << "aimpl: obj type id()" << std::endl;

		if(req_status(Req::Data) == ReqStatus::OK) {
			if(auto D = data(); D && *D) return (*D)->type_id();
		}
		return type_descriptor::nil().name;
	},

	// obtain inode
	[this](a_lnk_inode) -> result_or_errbox<inodeptr> {
		//pdbg() << "aimpl: inode()" << std::endl;

		return get_inode();
	},

	// default handler for `data_node` that works via `data`
	[this](a_lnk_dnode, bool wait_if_busy) -> result_or_errbox<sp_node> {
		//pdbg() << "aimpl: data_node() status = " <<
		//	(int)status_[0].value << (int)status_[1].value << std::endl;

		return data_node_ex(wait_if_busy);
	},

	// default `data` handler calls virtual `data()` function that by default returns nullptr
	[this](a_lnk_data, bool wait_if_busy) -> caf::result< result_or_errbox<sp_obj> > {
		//pdbg() << "aimpl: data() status =" <<
		//	(int)status_[0].value << (int)status_[1].value << std::endl;

		return data_node_ex(wait_if_busy);
	}
}; }

auto link_actor::make_simple_behavior() -> behavior_type {
	return caf::message_handler {
		// directly call `data()` without status changing
		[this](a_lnk_data, bool) -> result_or_errbox<sp_obj> {
			//pdbg() << "aimpl: simple data()" << std::endl;
			return data();
		},

		// same for `data_node`
		[this](a_lnk_dnode, bool) -> result_or_errbox<sp_node> {
			//pdbg() << "aimpl: simple data_node()" << std::endl;
			return data().map( [](sp_obj&& obj) -> sp_node {
				if(obj && obj->is_node()) return std::static_pointer_cast<tree::node>(std::move(obj));
				return nullptr;
			} );
		},

		// easier obj ID & object type id retrival
		[this](a_lnk_otid) -> std::string {
			//pdbg() << "aimpl: simple obj type id()" << std::endl;
			return data()
				.and_then([](const sp_obj& obj) -> result_or_errbox<std::string> {
					if(obj) return obj->type_id();
					return tl::make_unexpected(error::quiet(Error::EmptyData));
				})
				.value_or(type_descriptor::nil().name);
		},

		[this](a_lnk_oid) -> std::string {
			//pdbg() << "aimpl: simple obj id()" << std::endl;
			return data()
				.and_then([](const sp_obj& obj) -> result_or_errbox<std::string> {
					if(obj) return obj->id();
					return tl::make_unexpected(error::quiet(Error::EmptyData));
				})
				.value_or( to_string(boost::uuids::nil_uuid()) );
		}
	}.or_else(make_generic_behavior());
}

/*-----------------------------------------------------------------------------
 *  ilink
 *-----------------------------------------------------------------------------*/
ilink_actor::ilink_actor(caf::actor_config& cfg, std::string name, const sp_obj& data, Flags f)
	: super(cfg, std::move(name), f), inode_(make_inode(data))
{}

auto ilink_actor::get_inode() -> result_or_err<inodeptr> {
	return inode_;
};

auto ilink::pimpl() const -> ilink_actor* {
	return static_cast<ilink_actor*>(link::pimpl());
}

NAMESPACE_END(blue_sky::tree)
