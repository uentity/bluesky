/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief BS tree link class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "bare_link.h"
#include "engine.h"
#include "../atoms.h"
#include "../error.h"
#include "../propdict.h"

#include <caf/actor.hpp>
#include <caf/typed_actor.hpp>
#include <caf/scoped_actor.hpp>
#include <caf/group.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  base class of all links
 *-----------------------------------------------------------------------------*/
class BS_API link : public engine {
public:
	using bare_type = bare_link;
	using engine_impl = link_impl;
	using engine_actor = link_actor;
	using weak_ptr = engine::weak_ptr<link>;

	/// Interface of link actor, you can only send messages matching it
	using actor_type = caf::typed_actor<
		// get home group
		caf::replies_to<a_home>::with<caf::group>,
		// get pointee node group ID
		caf::replies_to<a_home_id>::with<std::string>,
		// get link ID
		caf::replies_to<a_lnk_id>::with<lid_type>,
		// get pointee OID
		caf::replies_to<a_lnk_oid>::with<std::string>,
		// get pointee type ID
		caf::replies_to<a_lnk_otid>::with<std::string>,

		// get link name
		caf::replies_to<a_lnk_name>::with<std::string>,
		// rename link, returns 1 on successfull rename
		caf::replies_to<a_lnk_rename, std::string>::with<std::size_t>,

		// get request status
		caf::replies_to<a_lnk_status, Req>::with<ReqStatus>,
		// reset request status
		caf::replies_to<a_lnk_status, Req, ReqReset, ReqStatus, ReqStatus>::with<ReqStatus>,

		// get link flags
		caf::replies_to<a_lnk_flags>::with<Flags>,
		// set link flags
		caf::reacts_to<a_lnk_flags, Flags>,

		// get inode
		caf::replies_to<a_lnk_inode>::with<result_or_errbox<inodeptr>>,
		// get data
		caf::replies_to<a_data, bool>::with<obj_or_errbox>,
		// get data node
		caf::replies_to<a_data_node, bool>::with<node_or_errbox>,
		// clone link
		caf::replies_to<a_clone, bool /* deep */>::with<link>,

		// run transaction in message queue of this link
		caf::replies_to<a_apply, link_transaction>::with<tr_result::box>,
		// run transaction in message queue of data object
		caf::replies_to<a_apply, a_data, obj_transaction>::with<tr_result::box>
	>;

	/// empty ctor will construct nil link
	link();
	/// convert from bare link
	explicit link(const bare_link& rhs);

	/// makes hard link
	link(std::string name, sp_obj data, Flags f = Plain);
	link(std::string name, node folder, Flags f = Plain);

	/// [NOTE] move sematics is doable but disabled (by explicit copy ctor)
	/// reason: moved from object MUST be immediately reset into nil link to not break invariants
	/// and that operation alone is equivalent to copy ctor/assignment call
	/// so, there's just no worth in it
	link(const link&) = default;
	auto operator=(const link&) -> link& = default;
	auto operator=(const bare_link&) -> link&;

	/// obtain bare link that provides direct access to link internals (passing over actor)
	/// [WANRING] main use case is for transactions and edge cases where yo now what you're doing
	auto bare() const -> bare_link;

	/// swap support
	friend auto swap(link& lhs, link& rhs) noexcept -> void {
		static_cast<engine&>(lhs).swap(rhs);
	}

	/// get typed actor of base link
	using engine::actor;
	auto actor() const -> actor_type {
		return engine::actor(*this);
	}

	/// if `deep` flag is set, then clone pointed object as well
	auto clone(bool deep = false) const -> link;

	/// create root link + propagate handle
	template<typename Link = link, typename... Args>
	static auto make_root(Args&&... args) -> Link {
		auto lnk = Link(std::forward<Args>(args)...);
		make_root_(lnk);
		return lnk;
	}

	/// test if link is nil
	auto is_nil() const -> bool;
	operator bool() const { return !is_nil(); }
	/// makes link nil
	auto reset() -> void;

	/// get link's container
	auto owner() const -> node;

	///////////////////////////////////////////////////////////////////////////////
	//  Fast link info requests
	//
	/// access link's unique ID
	auto id() const -> lid_type;

	/// obtain link's symbolic name
	auto name() const -> std::string;
	/// same as above, but returns name directly from stored member
	/// required by node
	auto name(unsafe_t) const -> std::string;

	auto flags() const -> Flags;
	auto set_flags(Flags new_flags) const -> void;

	/// rename link & notify owner node
	auto rename(std::string new_name) const -> bool;
	auto rename(launch_async_t, std::string new_name) const -> void;

	/// inspect object's inode
	auto info() const -> result_or_err<inode>;

	/// get link's object ID -- can return empty string
	auto oid() const -> std::string;

	/// get link's object type ID -- can return nil type ID
	auto obj_type_id() const -> std::string;

	/// applies functor to link atomically (invoke in link's queue)
	auto apply(link_transaction tr) const -> error;

	///////////////////////////////////////////////////////////////////////////////
	//  Pointee data API
	//
	/// get pointer to object link is pointing to -- slow, never returns invalid (NULL) sp_obj
	auto data_ex(bool wait_if_busy = true) const -> obj_or_err;
	/// simple data accessor that returns nullptr on error
	auto data() const -> sp_obj {
		return data_ex().value_or(nullptr);
	}
	/// directly return cached value (if any)
	auto data(unsafe_t) const -> sp_obj;

	/// return tree::node if contained object is a node -- slow, never returns invalid (NULL) sp_obj
	/// derived class can return cached node info
	auto data_node_ex(bool wait_if_busy = true) const -> node_or_err;
	/// simple tree::node accessor that returns nullptr on error
	auto data_node() const -> node;
	/// directly return cached value (if any)
	auto data_node(unsafe_t) const -> node;

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
	auto data_node_hid() const -> result_or_err<std::string>;

	/// make pointee data modification atomically
	auto data_apply(obj_transaction tr) const -> tr_result;

	/// sends empty transaction object to trigger `data modified` signal
	auto data_touch(tr_result tres = {}) const -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  Async API
	//
	/// obtain data in async manner passing it to callback
	using process_data_cb = std::function<void(obj_or_err, link)>;
	auto data(process_data_cb f, bool high_priority = false) const -> void;
	/// ... and data node
	using process_dnode_cb = std::function<void(node_or_err, link)>;
	auto data_node(process_dnode_cb f, bool high_priority = false) const -> void;

	auto apply(launch_async_t, link_transaction tr) const -> void;
	auto data_apply(launch_async_t, obj_transaction tr) const -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  Subscribe to link events
	//
	using event_handler = std::function< void(event) >;

	/// returns ID of suscriber that is required for unsubscribe
	auto subscribe(event_handler f, Event listen_to = Event::All) const -> std::uint64_t;
	/// unsubscribe handlers from self & whole subtree
	auto unsubscribe(deep_t) const -> void;
	using engine::unsubscribe;

protected:
	using engine::operator=;

	/// accept link impl and optionally start internal actor
	link(sp_engine_impl impl, bool start_actor = true);

	/// ensure that rhs type id matches requested, otherwise link is nil
	link(const link& rhs, std::string_view tgt_type_id);

	auto pimpl() const -> link_impl*;

	/// maually start internal actor (if not started already)
	auto start_engine() -> bool;

private:
	friend atomizer;
	friend weak_ptr;
	friend link_impl;
	friend link_actor;
	friend node_impl;

	link(engine&&);

	static auto make_root_(const link& donor) -> void;
};

/// handy aliases
using links_v = std::vector<link>;
using lids_v = std::vector<lid_type>;

/// checked link cast (without throwing `WrongLinkCast` exception)
template<typename DestLink>
auto link_cast(const link& rhs) -> std::optional<DestLink> {
	if(DestLink::type_id_() == rhs.type_id())
		return DestLink(rhs);
	return {};
}

/*-----------------------------------------------------------------------------
 *  hard link stores direct pointer to object
 *  multiple hard links can point to the same object
 *-----------------------------------------------------------------------------*/
class BS_API hard_link : public link {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = link;

	/// take an object that link will point to
	hard_link(std::string name, sp_obj data, Flags f = Plain);
	/// construct `objnode` instance internally with passed folder
	hard_link(std::string name, node folder, Flags f = Plain);
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

	/// return pointee
	auto target() const -> link_or_err;
	/// return pointee path (if `human_readable` is true, return link names in path)
	auto target_path(bool human_readable = false) const -> std::string;

private:
	/// empty ctor won't start internal actor
	sym_link();
};

NAMESPACE_END(blue_sky::tree)

// support for hashed container of links
STD_HASH_BS_LINK(link)
STD_HASH_BS_LINK(hard_link)
STD_HASH_BS_LINK(weak_link)
STD_HASH_BS_LINK(sym_link)
