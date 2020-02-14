/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief BS tree link class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../atoms.h"
#include "../error.h"
#include "../objbase.h"
#include "../propdict.h"
#include "common.h"

#include <cereal/access.hpp>

#include <caf/actor.hpp>
#include <caf/typed_actor.hpp>
#include <caf/scoped_actor.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  base class of all links
 *-----------------------------------------------------------------------------*/
class BS_API link {
	// serialization support
	friend class blue_sky::atomizer;
	// my private impl
	friend class link_impl;
	friend class link_actor;
	// full access for node
	friend class node;
	friend class node_impl;
	friend class node_actor;

public:
	/// Interface of link actor, you can only send messages matching it
	using actor_type = caf::typed_actor<
        // terminate actor
		caf::reacts_to<a_bye>,
        // get link ID
		caf::replies_to<a_lnk_id>::with<lid_type>,
        // get pointee OID
		caf::replies_to<a_lnk_oid>::with<std::string>,
        // get pointee type ID
		caf::replies_to<a_lnk_otid>::with<std::string>,
        // get pointee node group ID
		caf::replies_to<a_node_gid>::with<result_or_errbox<std::string>>,

        // get link name
		caf::replies_to<a_lnk_name>::with<std::string>,
        // rename link
		caf::reacts_to<a_lnk_rename, std::string, bool>,
        // ack rename
		caf::reacts_to<a_ack, a_lnk_rename, std::string, std::string>,

        // get request status
		caf::replies_to<a_lnk_status, Req>::with<ReqStatus>,
        // reset request status
		caf::replies_to<a_lnk_status, Req, ReqReset, ReqStatus, ReqStatus>::with<ReqStatus>,
        // request status change ack
		caf::reacts_to<a_ack, a_lnk_status, Req, ReqStatus, ReqStatus>,

        // get link flags
		caf::replies_to<a_lnk_flags>::with<Flags>,
        // set link flags
		caf::reacts_to<a_lnk_flags, Flags>,

        // get inode
		caf::replies_to<a_lnk_inode>::with<result_or_errbox<inodeptr>>,
        // get data
		caf::replies_to<a_lnk_data, bool>::with<result_or_errbox<sp_obj>>,
        // get data node
		caf::replies_to<a_lnk_dnode, bool>::with<result_or_errbox<sp_node>>,
        // modify data
		caf::replies_to<a_apply, data_modificator_f, bool>::with<error::box>
	>;

	/// handle that wraps strong ref to link's internal actor
	/// and terminates it on destruction
	struct actor_handle;

	/// empty ctor will construct nil link
	link();
	virtual ~link();

	/// copy ctor & assignment
	link(const link& rhs);
	auto operator=(const link& rhs) -> link&;

	/// [NOTE] move operations CAN be implemeted, but will be MORE expensive than copy
	/// Call `reset()` to make link nill explicitly

	/// makes link nil
	auto reset() -> void;

	/// links are compared by ID
	friend auto operator==(const link& lhs, const link& rhs) -> bool {
		return lhs.id() == rhs.id();
	}

	friend auto operator<(const link& lhs, const link& rhs) -> bool {
		return lhs.id() < rhs.id();
	}

	/// get link's typed actor handle
	template<typename Link>
	static auto actor(const Link& L) {
		return caf::actor_cast<typename Link::actor_type>(L.raw_actor());
	}

	/// get typed actor of base link
	auto actor() const -> actor_type {
		return caf::actor_cast<actor_type>(raw_actor());
	}

	/// get link's scoped actor that can be used to make direct requests to internal actor
	auto factor() const -> const caf::scoped_actor&;

	/// because we cannot make explicit copies of link
	/// we need a dedicated function to make links clones
	/// if `deep` flag is set, then clone pointed object as well
	auto clone(bool deep = false) const -> link;

	/// create root link + propagate handle
	template<typename Link, typename... Args>
	static auto make_root(Args&&... args) -> Link {
		if(auto lnk = Link(std::forward<Args>(args)...)) {
			static_cast<link&>(lnk).propagate_handle();
			return lnk;
		}
		return link{};
	}

	/// test if link is nil
	auto is_nil() const -> bool;
	operator bool() const { return !is_nil(); }

	///////////////////////////////////////////////////////////////////////////////
	//  Fast API
	//
	/// query what kind of link is this
	auto type_id() const -> std::string_view;

	/// access link's unique ID
	auto id() const -> lid_type;

	/// obtain link's symbolic name
	auto name() const -> std::string;
	/// same as above, but returns name directly from stored member
	/// can cause data race
	auto name(unsafe_t) const -> std::string;

	/// get link's container
	auto owner() const -> sp_node;

	auto flags() const -> Flags;
	auto flags(unsafe_t) const -> Flags;
	auto set_flags(Flags new_flags) const -> void;

	/// rename link & notify owner node
	auto rename(std::string new_name) const -> void;

	/// inspect object's inode
	auto info() const -> result_or_err<inode>;
	auto info(unsafe_t) const -> result_or_err<inode>;
	/// get link's object ID -- fast, can return empty string
	auto oid() const -> std::string;

	/// get link's object type ID -- fast, can return nil type ID
	auto obj_type_id() const -> std::string;

	/// get request status
	auto req_status(Req request) const -> ReqStatus;
	/// unconditional reset request status
	auto rs_reset(Req request, ReqStatus new_status = ReqStatus::Void) const -> ReqStatus;
	/// conditional reset request status
	auto rs_reset_if_eq(Req request , ReqStatus self_rs, ReqStatus new_rs = ReqStatus::Void) const -> ReqStatus;
	auto rs_reset_if_neq(Req request, ReqStatus self_rs, ReqStatus new_rs = ReqStatus::Void) const -> ReqStatus;

	/// methods below are efficient checks that won't call `data_node()` if possible
	/// check if pointee is a node
	auto is_node() const -> bool;
	/// if pointee is a node, return node's actor group ID
	auto data_node_gid() const -> result_or_err<std::string>;

	///////////////////////////////////////////////////////////////////////////////
	//  Pointee data API
	//
	/// get pointer to object link is pointing to -- slow, never returns invalid (NULL) sp_obj
	auto data_ex(bool wait_if_busy = true) const -> result_or_err<sp_obj>;
	/// simple data accessor that returns nullptr on error
	auto data() const -> sp_obj {
		return data_ex().value_or(nullptr);
	}

	/// return tree::node if contained object is a node -- slow, never returns invalid (NULL) sp_obj
	/// derived class can return cached node info
	auto data_node_ex(bool wait_if_busy = true) const -> result_or_err<sp_node>;
	/// simple tree::node accessor that returns nullptr on error
	auto data_node() const -> sp_node {
		return data_node_ex().value_or(nullptr);
	}

	/// make pointee data modification atomically
	auto modify_data(data_modificator_f m, bool silent = false) const -> error;
	auto modify_data(launch_async_t, data_modificator_f m, bool silent = false) const -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  Async API
	//
	/// obtain data in async manner passing it to callback
	using process_data_cb = std::function<void(result_or_err<sp_obj>, link)>;
	auto data(process_data_cb f, bool high_priority = false) const -> void;
	/// ... and data node
	auto data_node(process_data_cb f, bool high_priority = false) const -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  Subscribe to link events
	//
	using handle_event_cb = std::function< void(link, Event, prop::propdict) >;

	/// returns ID of suscriber that is required for unsubscribe
	auto subscribe(handle_event_cb f, Event listen_to = Event::All) const -> std::uint64_t;
	auto unsubscribe(std::uint64_t event_cb_id) const -> void;

protected:
	/// accept link impl and optionally start internal actor
	/// if `self_tid` specified - ensure that `impl` is of requested type
	link(std::shared_ptr<link_impl> impl, bool start_actor = true);

	/// ensure that rhs type id matches requested, otherwise link is nil
	link(const link& rhs, std::string_view rhs_type_id);

	/// get access to link's impl for derived links
	auto pimpl() const -> link_impl*;

	/// return link's raw (dynamic-typed) actor handle
	auto raw_actor() const -> const caf::actor&;

	/// maually start internal actor (if not started already)
	auto start_engine() -> bool;

	/// set handle of passed node to self
	auto self_handle_node(const sp_node& N) const -> void;

	// silent replace old name with new in link's internals
	auto rename_silent(std::string new_name) const -> void;

	/// switch link's owner
	auto reset_owner(const sp_node& new_owner) const -> void;

	// if pointee is a node - set node's handle to self and return pointee
	// default implementation obtains node via `data_node_ex()` and sets it's handle to self
	// but derived link can change default behaviour
	auto propagate_handle() const -> result_or_err<sp_node>;

private:
	// scoped actor for requests
	const caf::scoped_actor factor_;

	// strong ref to internal link's actor
	// [NOTE] trick with shared ptr to handle is required to correctly track `link` instances
	// and terminate internal actor when no more links exist
	std::shared_ptr<actor_handle> actor_;

	// string ref to link's impl
	friend class link_impl;
	std::shared_ptr<link_impl> pimpl_;
};

/// handy aliases
using links_v = std::vector<link>;
using lids_v = std::vector<lid_type>;

using cached_link_actor_type = link::actor_type::extend<
	// get data cache
	caf::replies_to<a_lnk_dcache>::with<sp_obj>
>;

/*-----------------------------------------------------------------------------
 *  hard link stores direct pointer to object
 *  multiple hard links can point to the same object
 *-----------------------------------------------------------------------------*/
class BS_API hard_link : public link {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = link;
	using actor_type = cached_link_actor_type;

	/// ctor -- additionaly accepts a pointer to object
	hard_link(std::string name, sp_obj data, Flags f = Plain);
	/// convert from base link
	hard_link(const link& rhs);

	static auto type_id_() -> std::string_view;

private:
	/// empty ctor won't start internal actor
	hard_link();
};

/*-----------------------------------------------------------------------------
 *  weak link is same as hard link, but stores weak link to data
 *  intended to be used to add class memebers self tree structure
 *-----------------------------------------------------------------------------*/
class BS_API weak_link : public link {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = link;
	using actor_type = cached_link_actor_type;

	/// ctor -- additionaly accepts a pointer to object
	weak_link(std::string name, const sp_obj& data, Flags f = Plain);
	/// convert from base link
	weak_link(const link& rhs);

	static auto type_id_() -> std::string_view;

private:
	/// empty ctor won't start internal actor
	weak_link();
};

/*-----------------------------------------------------------------------------
 *  symbolic link is actually a link to another link, which is specified
 *  as absolute or relative string path
 *-----------------------------------------------------------------------------*/
class BS_API sym_link : public link {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = link;

	/// ctor -- pointee is specified by string path
	sym_link(std::string name, std::string path, Flags f = Plain);
	/// ctor -- pointee is specified directly - absolute path will be stored
	sym_link(std::string name, const link& src, Flags f = Plain);
	/// convert from base link
	sym_link(const link& rhs);

	static auto type_id_() -> std::string_view;

	/// additional sym link API
	/// check is pointed link is alive, sets Data status to proper state
	auto check_alive() -> bool;

	/// return stored pointee path
	auto src_path(bool human_readable = false) const -> std::string;

private:
	/// empty ctor won't start internal actor
	sym_link();
};

NAMESPACE_END(blue_sky::tree)
