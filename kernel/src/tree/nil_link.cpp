/// @file
/// @author uentity
/// @date 06.02.2020
/// @brief Nil (invalid) link impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "nil_link.h"
#include "link_actor.h"

#include <bs/defaults.h>
#include <bs/kernel/radio.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

NAMESPACE_BEGIN(blue_sky::tree)
///////////////////////////////////////////////////////////////////////////////
//  nil link actor
//
struct nil_link::self_actor : caf::event_based_actor {
	using super = caf::event_based_actor;

	self_actor(caf::actor_config& cfg)
		: super(cfg)
	{}

	auto make_behavior() -> behavior_type override { return link::actor_type::behavior_type {
	
		[=](a_home) -> caf::group { return {}; },
		[=](a_lnk_id) -> lid_type { return nil_uid; },
		[=](a_lnk_oid) -> std::string { return nil_oid; },
		[=](a_lnk_otid) -> std::string { return nil_otid; },
		[=](a_node_gid) -> result_or_errbox<std::string> {
			return tl::make_unexpected(error{Error::EmptyData});
		},

		// deny rename
		[=](a_lnk_name) -> std::string { return defaults::tree::nil_link_name; },
		[=](a_lnk_rename, const std::string&, bool) -> void {},

		// status alwaye Void
		[=](a_lnk_status, Req) -> ReqStatus { return ReqStatus::Void; },
		[=](a_lnk_status, Req, ReqReset, ReqStatus, ReqStatus) -> ReqStatus {
			return ReqStatus::Void;
		},

		[=](a_lnk_flags) { return Flags::Plain; },
		[=](a_lnk_flags, Flags) {},

		// all data is null
		[=](a_lnk_inode) -> result_or_errbox<inodeptr> {
			return tl::make_unexpected(error{Error::EmptyInode});
		},
		[=](a_lnk_data, bool) -> result_or_errbox<sp_obj> {
			return tl::make_unexpected(error{Error::EmptyData});
		},
		[=](a_lnk_dnode, bool) -> result_or_errbox<sp_node> {
			return tl::make_unexpected(error{Error::EmptyData});
		}

	}.unbox(); }

	auto on_exit() -> void override {
		nil_link::reset();
	}
};

///////////////////////////////////////////////////////////////////////////////
//  nil link impl
//
struct nil_link::self_impl : link_impl {
	using super = link_impl;
	using super::super;

	// return global instance of nil link inside optional to destroy at any moment
	using internals_t = std::pair<sp_limpl, sp_ahandle>;
	static auto internals() -> internals_t& {
		static auto self_ = internals_t(
			std::make_shared<nil_link::self_impl>(),
			std::make_shared<link::actor_handle>( kernel::radio::system().spawn<nil_link::self_actor>() )
		);
		return self_;
	}

	// always return same actor from internals
	auto spawn_actor(sp_limpl) const -> caf::actor override {
		return internals().second->actor_;
	}

	auto clone(bool deep) const -> sp_limpl override {
		return nullptr;
	}

	auto data() -> result_or_err<sp_obj> override {
		return tl::make_unexpected(Error::EmptyData);
	}

	self_impl()
		: super(defaults::tree::nil_link_name, Flags::Plain)
	{
		id_ = nil_uid;
	}

	LIMPL_TYPE_DECL
};

LIMPL_TYPE_DEF(nil_link::self_impl, "__nil_link__")

///////////////////////////////////////////////////////////////////////////////
//  nil link
//
auto nil_link::reset() -> void {
	// explicitly reset signleton nil link acctor handle & internals
	auto& [nl_impl, nl_actor] = nil_link::self_impl::internals();
	nl_actor->actor_ = nullptr;
	nl_actor.reset();
	nl_impl.reset();
}

auto nil_link::stop(bool wait_exit) -> void {
	auto& [_, nil_handle] = nil_link::self_impl::internals();
	if(nil_handle) {
		auto nil_actor = nil_handle->actor_;
		auto waiter = caf::scoped_actor{KRADIO.system(), false};
		waiter->send_exit(nil_actor, caf::exit_reason::kill);
		if(wait_exit)
			waiter->wait_for(nil_actor);
	}
}

auto nil_link::pimpl() -> const sp_limpl& { return nil_link::self_impl::internals().first; }

auto nil_link::actor() -> const sp_ahandle& { return nil_link::self_impl::internals().second; }

NAMESPACE_END(blue_sky::tree)
