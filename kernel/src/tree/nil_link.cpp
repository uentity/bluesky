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

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
NAMESPACE_BEGIN()
///////////////////////////////////////////////////////////////////////////////
//  nil link actor
//
struct BS_HIDDEN_API nil_link_actor : link_actor {
	using link_actor::link_actor;

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
		[=](a_lnk_rename, std::string, bool) -> void {},

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
		return spawn_lactor<nil_link_actor, caf::spawn_options::hide_flag>(std::move(limpl));
	}

	auto clone(bool deep) const -> sp_limpl override {
		return nullptr;
	}

	auto data() -> result_or_err<sp_obj> override {
		return tl::make_unexpected(Error::EmptyData);
	}

	auto type_id() const -> std::string_view override { return type_id_(); }

	static auto type_id_() -> std::string_view {
		// generate random type ID (UUID string)
		static const auto session_nil = []() -> std::string_view {
			static auto buf = std::array<char, 36>{}; // length of UUID string
			auto x = to_string(boost::uuids::random_generator()());
			std::copy(x.begin(), x.end(), buf.begin());
			return { buf.data(), buf.size() };
		}();

		return session_nil;
	}
};

NAMESPACE_END()

///////////////////////////////////////////////////////////////////////////////
//  nil link
//
nil_link::nil_link(sp_limpl pimpl) :
	pimpl_(pimpl),
	// auto start nil link service
	actor_( std::make_shared<link::actor_handle>(pimpl_->spawn_actor(pimpl_)) )
{}

auto nil_link::self() -> nil_link& {
	static auto self_ = nil_link{ std::make_shared<nil_link_impl>() };
	return self_;
}

auto nil_link::stop() -> void {
	self().pimpl_.reset();
	self().actor_.reset();
}

nil_link::~nil_link() {
	stop();
}

auto nil_link::pimpl() -> const sp_limpl& { return self().pimpl_; }

auto nil_link::actor() -> const sp_ahandle& { return self().actor_; }

NAMESPACE_END(blue_sky::tree)
