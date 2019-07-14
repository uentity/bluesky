/// @file
/// @author uentity
/// @date 08.07.2019
/// @brief LInk's async actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/link.h>
#include <bs/tree/node.h>
#include <bs/kernel/config.h>

#include "actor_common.h"
#include "link_invoke.h"

#include <boost/uuid/uuid_io.hpp>

#include <caf/actor_system.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/actor_ostream.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
using namespace tree::detail;

using id_type = link::id_type;
using Flags = link::Flags;

/*-----------------------------------------------------------------------------
 *  link_actor
 *-----------------------------------------------------------------------------*/
class BS_API link_actor : public caf::event_based_actor {
public:
	//using super = caf::stateful_actor<lnk_state>;
	using super = caf::event_based_actor;
	using behavior_type = super::behavior_type;

	id_type id_;
	std::string name_;
	Flags flags_;
	/// owner node
	std::weak_ptr<tree::node> owner_;
	/// status of operations
	status_handle status_[2];
	// sync access
	std::mutex solo_;

	// timeout for most queries
	timespan timeout_;
	// sender used for blocking wait for responses
	caf::function_view<caf::actor> factor_;

	// keep local link group
	caf::group self_grp;

	// store cached data here
	//sp_obj cache_;

	// [DEBUG]
	auto pdbg() -> caf::actor_ostream;

	link_actor(caf::actor_config& cfg, std::string name, Flags f, timespan data_timeout = def_data_timeout);
	// virtual dtor
	virtual ~link_actor();

	// cleanup code executes leaving from local group & must be called from outside
	auto goodbye() -> void;

	// get handle of this actor
	inline auto handle() -> caf::actor {
		return caf::actor_cast<caf::actor>(address());
	}

	auto req_status(Req request) const -> ReqStatus;

	auto rs_reset(Req request, ReqStatus new_rs, bool silent = false) -> ReqStatus;

	auto rs_reset_if_eq(Req request, ReqStatus self_rs, ReqStatus new_rs, bool silent = false) -> ReqStatus;

	auto rs_reset_if_neq(Req request, ReqStatus self_rs, ReqStatus new_rs, bool silent = false) -> ReqStatus;

	auto reset_owner(const sp_node& new_owner) -> void;

	/// get pointer to object link is pointing to -- slow, never returns invalid (NULL) sp_obj
	auto data_ex(bool wait_if_busy = true) -> result_or_err<sp_obj>;

	/// return tree::node if contained object is a node -- slow, never returns invalid (NULL) sp_obj
	/// derived class can return cached node info
	auto data_node_ex(bool wait_if_busy = true) -> result_or_err<sp_node>;

	/// obtain inode pointer
	/// default impl do it via `data_ex()` call
	virtual auto get_inode() -> result_or_err<inodeptr>;

	/// create or set or create inode for given target object
	/// [NOTE] if `new_info` is non-null, returned inode may be NOT EQUAL to `new_info`
	static auto make_inode(const sp_obj& target, inodeptr new_info = nullptr) -> inodeptr;

	///////////////////////////////////////////////////////////////////////////////
	//  behavior
	//
	// returns generic behavior
	auto make_behavior() -> behavior_type override;

	// Generic impl of link's beahvior suitable for all link types
	auto make_generic_behavior() -> behavior_type;

	// Difference from original is that `data` and `data_node` handlers directly invoke
	// virtual `data()` and `data_node()` methods and don't touch statuses at all.
	// Such behavior is suitable for base links that directly contain data with always quick access
	auto make_simple_behavior() -> behavior_type;

protected:
	/// [NOTE] download pointee data - must be implemented by derived links
	virtual auto data() -> result_or_err<sp_obj> = 0;

	// [NOTE] derived links can choose to override this dumb impl
	virtual auto data_node() -> result_or_err<sp_node>;
};

/// spawn derived actor
template<typename T, caf::spawn_options Os = caf::no_spawn_options, class... Ts>
inline auto spawn_lactor(Ts&&... args) {
	// spawn actor
	auto A = kernel::config::actor_system().spawn<T, Os>(std::forward<Ts>(args)...);
	auto L = caf::actor_cast<T*>(A);
	if(!L) throw error{ "Cannot spawn link actor!" };
	return A;
}

/*-----------------------------------------------------------------------------
 *  derived links actors
 *-----------------------------------------------------------------------------*/
struct BS_API ilink_actor : public link_actor {
	// ilink carries inode
	inodeptr inode_;

	using super = link_actor;

	ilink_actor(caf::actor_config& cfg, std::string name, const sp_obj& data, Flags f);

	// returns stored pointer
	auto get_inode() -> result_or_err<inodeptr> override final;
};

struct BS_API hard_link_actor : public ilink_actor {
	sp_obj data_;

	using super = ilink_actor;
	
	hard_link_actor(caf::actor_config& cfg, std::string name, sp_obj data, Flags f);

	auto data() -> result_or_err<sp_obj> override;

	// installs simple behavior
	auto make_behavior() -> behavior_type override;
};

struct BS_API weak_link_actor : public ilink_actor {
	std::weak_ptr<objbase> data_;

	using super = ilink_actor;
	
	weak_link_actor(caf::actor_config& cfg, std::string name, const sp_obj& data, Flags f);

	auto data() -> result_or_err<sp_obj> override;

	// installs simple behavior
	auto make_behavior() -> behavior_type override;
};

struct BS_API sym_link_actor : public link_actor {
	std::string path_;

	using super = link_actor;
	using super::owner_;
	
	sym_link_actor(caf::actor_config& cfg, std::string name, std::string path, Flags f);

	auto data() -> result_or_err<sp_obj> override;
};

NAMESPACE_END(blue_sky::tree)
