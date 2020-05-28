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
NAMESPACE_BEGIN()

///////////////////////////////////////////////////////////////////////////////
//  nil link actor
//
struct BS_HIDDEN_API nil_link_actor : link_actor {
	using super = link_actor;

	nil_link_actor(caf::actor_config& cfg, caf::group home, sp_limpl Limpl)
		: super(cfg, std::move(home), std::move(Limpl))
	{
		// join kernel's group
		join(KRADIO.khome());
	}

	auto data_ex(obj_processor_f f, ReqOpts) -> void override {
		using R = result_or_errbox<sp_obj>;
		f(R{ tl::unexpect, error{Error::EmptyData} });
	}

	auto data_node_ex(node_processor_f f, ReqOpts) -> void override {
		using R = result_or_errbox<sp_node>;
		f(R{ tl::unexpect, error{Error::EmptyData} });
	}

	auto make_behavior() -> behavior_type override { return caf::message_handler{
		[=](a_lnk_oid) -> std::string { return nil_oid; },
		[=](a_lnk_otid) -> std::string { return nil_otid; },

		// deny rename
		[=](a_lnk_rename, const std::string&, bool) -> void {},

		// status alwaye Error
		[=](a_lnk_status, Req, ReqReset, ReqStatus, ReqStatus) -> ReqStatus {
			return ReqStatus::Error;
		},

		// all data is null
		[=](a_lnk_inode) -> result_or_errbox<inodeptr> {
			return tl::make_unexpected(error{Error::EmptyInode});
		},
		[=](a_lnk_data, bool) -> caf::result<result_or_errbox<sp_obj>> {
			return tl::make_unexpected(error{Error::EmptyData});
		},
		[=](a_lnk_dnode, bool) -> caf::result<result_or_errbox<sp_node>> {
			return tl::make_unexpected(error{Error::EmptyData});
		}

	}.or_else(link_actor::make_behavior()); }

	auto on_exit() -> void override {
		link_actor::on_exit();
		nil_link::stop();
	}
};

///////////////////////////////////////////////////////////////////////////////
//  nil link impl
//
struct nil_link_impl : link_impl {
	using super = link_impl;
	using super::super;

	nil_link_impl()
		: super(defaults::tree::nil_link_name, Flags::Plain)
	{
		id_ = nil_uid;
	}

	auto spawn_actor(sp_limpl limpl) const -> caf::actor override {
		// spawn actor with nil home group
		return kernel::radio::system().spawn<nil_link_actor>(caf::group{}, std::move(limpl));
	}

	auto clone(bool deep) const -> sp_limpl override {
		return nullptr;
	}

	auto data() -> result_or_err<sp_obj> override {
		return tl::make_unexpected(Error::EmptyData);
	}

	LIMPL_TYPE_DECL
};

LIMPL_TYPE_DEF(nil_link_impl, "__nil_link__")

// return global instance of nil link inside optional to destroy at any moment
auto nil_link_internals() -> std::pair<sp_limpl, sp_ahandle>& {
	static auto self_ = [] {
		sp_limpl nl_impl = std::make_shared<nil_link_impl>();
		auto nl_actor = std::make_shared<link::actor_handle>(nl_impl->spawn_actor(nl_impl));
		return std::make_pair(std::move(nl_impl), std::move(nl_actor));
	}();
	return self_;
}

NAMESPACE_END()

///////////////////////////////////////////////////////////////////////////////
//  nil link
//
auto nil_link::stop() -> void {
	// explicitly reset signleton nil link acctor handle & internals
	auto& [nl_impl, nl_actor] = nil_link_internals();
	nl_actor->actor_ = nullptr;
	nl_actor.reset();
	nl_impl.reset();
}

auto nil_link::pimpl() -> const sp_limpl& { return nil_link_internals().first; }

auto nil_link::actor() -> const sp_ahandle& { return nil_link_internals().second; }

NAMESPACE_END(blue_sky::tree)
