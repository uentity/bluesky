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
#include "../detail/enumops.h"
#include "../propdict.h"
#include "inode.h"

#include <boost/uuid/uuid.hpp>
#include <cereal/access.hpp>

#include <caf/actor.hpp>
#include <caf/typed_actor.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

// denote possible tree events
enum class Event : std::uint64_t {
	LinkRenamed = 1,
	LinkStatusChanged = 2,
	LinkInserted = 4,
	LinkErased = 8,
	LinkDeleted = 16,
	All = std::uint64_t(-1)
};

// denote that we don't want to wait until invoke result is available
struct launch_async_t {};
inline constexpr auto launch_async = launch_async_t{};

// denote operation that is thread-unsafe and can cause data race
struct unsafe_t {};
inline constexpr auto unsafe = unsafe_t{};

class link_impl;
class node_impl;

/*-----------------------------------------------------------------------------
 *  base class of all links
 *-----------------------------------------------------------------------------*/
class BS_API link  : public std::enable_shared_from_this<link> {
	// serialization support
	friend class blue_sky::atomizer;
	// my private impl
	friend class link_impl;
	// full access for node
	friend class node;
	friend class node_impl;

public:
	///////////////////////////////////////////////////////////////////////////////
	//  static types, enumes, typed actor interface for link
	//
	using id_type = boost::uuids::uuid;
	using sp_link = std::shared_ptr<link>;
	using sp_clink = std::shared_ptr<const link>;
	using link_ptr = object_ptr<link>;
	using clink_ptr = object_ptr<const link>;

	/// object data requests
	enum class Req { Data = 0, DataNode = 1 };
	/// states of reuqest
	enum class ReqStatus { Void, Busy, OK, Error };
	/// request status reset conditions
	enum class ReqReset {
		Always = 0, IfEq = 1, IfNeq = 2
	};

	/// flags reflect link properties and state
	enum Flags {
		Plain = 0,
		Persistent = 1,
		Disabled = 2,
		LazyLoad = 4
	};

	using modificator_f = std::function< error(sp_obj) >;

	/// Interface of link actor, you can only send messages matching it
	using actor_type = caf::typed_actor<
        // terminate actor
		caf::reacts_to<a_bye>,
        // get link ID
		caf::replies_to<a_lnk_id>::with<id_type>,
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
		caf::replies_to<a_apply, modificator_f, bool>::with<error::box>
	>;

	// extract actor type from link
	template<typename Link>
	static auto actor(const Link& L) {
		return L.template actor<typename Link::actor_type>();
	}

	/// provide shared pointers casted to derived type
	template< class Derived >
	decltype(auto) bs_shared_this() const {
		return std::static_pointer_cast< const Derived, const link >(this->shared_from_this());
	}

	template< class Derived >
	decltype(auto) bs_shared_this() {
		return std::static_pointer_cast< Derived, link >(this->shared_from_this());
	}

	virtual ~link();

	///////////////////////////////////////////////////////////////////////////////
	//  Generate
	//
	/// because we cannot make explicit copies of link
	/// we need a dedicated function to make links clones
	/// if `deep` flag is set, then clone pointed object as well
	virtual auto clone(bool deep = false) const -> sp_link = 0;

	/// create root link to node with and set it as node's handle
	template<typename Link, typename... Args>
	static auto make_root(Args&&... args) -> std::shared_ptr<Link> {
		if(auto lnk = std::make_shared<Link>(std::forward<Args>(args)...)) {
			static_cast<link*>(lnk.get())->propagate_handle();
			return lnk;
		}
		return nullptr;
	}

	///////////////////////////////////////////////////////////////////////////////
	//  Always fast API
	//
	/// query what kind of link is this
	virtual auto type_id() const -> std::string = 0;

	/// access link's unique ID
	auto id() const -> id_type;

	/// obtain link's symbolic name
	auto name() const -> std::string;
	/// same as above, but returns name directly from stored member
	/// can cause data race
	auto name(unsafe_t) const -> std::string;

	/// get link's container
	auto owner() const -> sp_node;

	auto flags() const -> Flags;
	auto set_flags(Flags new_flags) -> void;

	/// rename link & notify owner node
	auto rename(std::string new_name) -> void;

	/// inspect object's inode
	auto info() const -> result_or_err<inode>;
	/// get link's object ID -- fast, can return empty string
	auto oid() const -> std::string;

	/// get link's object type ID -- fast, can return nil type ID
	auto obj_type_id() const -> std::string;

	/// get request status
	auto req_status(Req request) const -> ReqStatus;
	/// unconditional reset request status
	auto rs_reset(Req request, ReqStatus new_status = ReqStatus::Void) -> ReqStatus;
	/// conditional reset request status
	auto rs_reset_if_eq(Req request , ReqStatus self_rs, ReqStatus new_rs = ReqStatus::Void) -> ReqStatus;
	auto rs_reset_if_neq(Req request, ReqStatus self_rs, ReqStatus new_rs = ReqStatus::Void) -> ReqStatus;

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
	auto modify_data(modificator_f m, bool silent = false) const -> error;
	auto modify_data(launch_async_t, modificator_f m, bool silent = false) const -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  Async API
	//
	/// obtain data in async manner passing it to callback
	using process_data_cb = std::function<void(result_or_err<sp_obj>, sp_clink)>;
	auto data(process_data_cb f, bool high_priority = false) const -> void;
	/// ... and data node
	auto data_node(process_data_cb f, bool high_priority = false) const -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  Subscribe to link events
	//
	using handle_event_cb = std::function< void(sp_clink, Event, prop::propdict) >;

	/// returns ID of suscriber that is required for unsubscribe
	auto subscribe(handle_event_cb f, Event listen_to = Event::All) const -> std::uint64_t;
	auto unsubscribe(std::uint64_t event_cb_id) const -> void;

protected:
	/// accept link impl and optionally start internal actor
	link(std::shared_ptr<link_impl> impl, bool start_actor = true);

	/// deny making link copies
	link(const link&) = delete;

	/// return link's typed actor handle
	template<typename ActorType = actor_type>
	auto actor() const {
		return caf::actor_cast<ActorType>(actor_);
	}

	/// maually start internal actor (if not started already)
	auto start_engine() -> bool;

	// silent replace old name with new in link's internals
	auto rename_silent(std::string new_name) -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  misc API for inherited links
	//
	/// switch link's owner
	virtual auto reset_owner(const sp_node& new_owner) -> void;

	// if pointee is a node - set node's handle to self and return pointee
	// default implementation obtains node via `data_node_ex()` and sets it's handle to self
	// but derived link can change default behaviour
	virtual auto propagate_handle() -> result_or_err<sp_node>;
	/// set handle of passed node to self
	auto self_handle_node(const sp_node& N) -> void;

	// get access to link's impl for derived links
	auto pimpl() const -> link_impl*;

	// strong ref to internal typeless link's actor
	caf::actor actor_;

private:
	// string ref to link's impl
	friend class link_impl;
	std::shared_ptr<link_impl> pimpl_;
};
using sp_link = link::sp_link;
using sp_clink = link::sp_clink;
using link_ptr = link::link_ptr;
using clink_ptr = link::clink_ptr;

using cached_link_actor_type = link::actor_type::extend<
	// get data cache
	caf::replies_to<a_lnk_dcache>::with<sp_obj>
>;

///////////////////////////////////////////////////////////////////////////////
//  link with bundled inode
//
struct ilink_impl;
class BS_API ilink : public link {
	friend class blue_sky::atomizer;

protected:
	using super = link;
	using super::super;

	auto pimpl() const -> ilink_impl*;
};

/*-----------------------------------------------------------------------------
 *  hard link stores direct pointer to object
 *  multiple hard links can point to the same object
 *-----------------------------------------------------------------------------*/
struct hard_link_impl;
class BS_API hard_link : public ilink {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = ilink;
	using actor_type = cached_link_actor_type;

	/// ctor -- additionaly accepts a pointer to object
	hard_link(std::string name, sp_obj data, Flags f = Plain);

	/// implement link's API
	auto clone(bool deep = false) const -> sp_link override;

	auto type_id() const -> std::string override;

private:
	/// empty ctor won't start internal actor
	hard_link();

	auto pimpl() const -> hard_link_impl*;
};

/*-----------------------------------------------------------------------------
 *  weak link is same as hard link, but stores weak link to data
 *  intended to be used to add class memebers self tree structure
 *-----------------------------------------------------------------------------*/
struct weak_link_impl;
class BS_API weak_link : public ilink {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = ilink;
	using actor_type = cached_link_actor_type;

	/// ctor -- additionaly accepts a pointer to object
	weak_link(std::string name, const sp_obj& data, Flags f = Plain);

	/// implement link's API
	auto clone(bool deep = false) const -> sp_link override;

	auto type_id() const -> std::string override;

private:
	/// empty ctor won't start internal actor
	weak_link();

	auto pimpl() const -> weak_link_impl*;
	auto propagate_handle() -> result_or_err<sp_node> override;
};

/*-----------------------------------------------------------------------------
 *  symbolic link is actually a link to another link, which is specified
 *  as absolute or relative string path
 *-----------------------------------------------------------------------------*/
struct sym_link_impl;
class BS_API sym_link : public link {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = link;

	/// ctor -- pointee is specified by string path
	sym_link(std::string name, std::string path, Flags f = Plain);
	/// ctor -- pointee is specified directly - absolute path will be stored
	sym_link(std::string name, const sp_link& src, Flags f = Plain);

	/// implement link's API
	auto clone(bool deep = false) const -> sp_link override;

	auto type_id() const -> std::string override;

	/// additional sym link API
	/// check is pointed link is alive, sets Data status to proper state
	auto check_alive() -> bool;

	/// return stored pointee path
	auto src_path(bool human_readable = false) const -> std::string;

private:
	/// empty ctor won't start internal actor
	sym_link();

	auto pimpl() const -> sym_link_impl*;
	auto reset_owner(const sp_node& new_owner) -> void override;

	auto propagate_handle() -> result_or_err<sp_node> override;
};

NAMESPACE_END(blue_sky::tree)

BS_ALLOW_ENUMOPS(blue_sky::tree::Event)
