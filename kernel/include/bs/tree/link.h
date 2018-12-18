/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief BS tree link class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once
#include "../error.h"
#include "../objbase.h"
#include "../detail/enumops.h"
#include "inode.h"

#include <atomic>
#include <chrono>
#include <boost/uuid/uuid.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  base class of all links
 *-----------------------------------------------------------------------------*/
class BS_API link  : public std::enable_shared_from_this<link> {
public:
	using id_type = boost::uuids::uuid;
	using sp_link = std::shared_ptr<link>;
	using sp_clink = std::shared_ptr<const link>;

	/// virtual dtor
	virtual ~link();

	/// provide shared pointers casted to derived type
	template< class Derived >
	decltype(auto) bs_shared_this() const {
		return std::static_pointer_cast< const Derived, const link >(this->shared_from_this());
	}

	template< class Derived >
	decltype(auto) bs_shared_this() {
		return std::static_pointer_cast< Derived, link >(this->shared_from_this());
	}

	/// access link's unique ID
	auto id() const -> const id_type&;

	/// obtain link's symbolic name
	auto name() const -> std::string;

	/// get link's container
	auto owner() const -> sp_node;

	/// flags reflect link properties and state
	enum Flags {
		Plain = 0,
		Persistent = 1,
		Disabled = 2,
		LazyLoad = 4
	};
	auto flags() const -> Flags;
	auto set_flags(Flags new_flags) -> void;

	/// rename link & notify owner node
	auto rename(std::string new_name) -> void;

	/// inspect object's inode
	auto info() const -> result_or_err<inode>;

	/// because we cannot make explicit copies of link
	/// we need a dedicated function to make links clones
	/// if `deep` flag is set, then clone pointed object as well
	virtual auto clone(bool deep = false) const -> sp_link = 0;

	/// create root link to node with handle preset to returned link
	template<typename Link, typename... Args>
	static auto make_root(Args&&... args) -> std::shared_ptr<Link> {
		if(auto lnk = std::make_shared<Link>(std::forward<Args>(args)...)) {
			static_cast<link*>(lnk.get())->propagate_handle();
			return lnk;
		}
		return nullptr;
	}

	/// query what kind of link is this
	virtual auto type_id() const -> std::string = 0;

	///////////////////////////////////////////////////////////////////////////////
	//  sync API
	//
	/// get link's object ID -- fast, can return empty string
	virtual auto oid() const -> std::string;

	/// get link's object type ID -- fast, can return nil type ID
	virtual auto obj_type_id() const -> std::string;

	/// get pointer to object link is pointing to -- slow, never returns invalid (NULL) sp_obj
	/// NOTE: returned pointer can be null
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

	///////////////////////////////////////////////////////////////////////////////
	//  async API
	//
	// enum slow requests
	enum class Req { Data = 0, DataNode = 1 };
	// states of single reuqest
	enum class ReqStatus { Void, Busy, OK, Error };

	// get/set request status
	auto req_status(Req request) const -> ReqStatus;
	// unconditional reset request status
	auto rs_reset(Req request, ReqStatus new_status = ReqStatus::Void) const -> ReqStatus;
	auto rs_reset_if_eq(Req request , ReqStatus self_rs, ReqStatus new_rs = ReqStatus::Void) const -> ReqStatus;
	auto rs_reset_if_neq(Req request, ReqStatus self_rs, ReqStatus new_rs = ReqStatus::Void) const -> ReqStatus;

	/// obtain data in async manner passing it to callback
	using process_data_cb = std::function<void(result_or_err<sp_obj>, sp_clink)>;
	auto data(process_data_cb f, bool high_priority = false) const -> void;
	/// ... and data node
	auto data_node(process_data_cb f, bool high_priority = false) const -> void;

protected:
	// serialization support
	friend class blue_sky::atomizer;
	// full access for node
	friend class node;

	/// ctor accept name of created link
	link(std::string name, Flags f = Plain);

	/// deby link copying
	link(const link&) = delete;

	// silent replace old name with new in link's internals
	auto rename_silent(std::string new_name) -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  sync data API
	//  Implementation need not to bother with reuqest status
	//  Performance hint: you can check if status is OK
	//  to omit useless repeating possibly long operation calls
	//

	/// switch link's owner
	virtual auto reset_owner(const sp_node& new_owner) -> void;

	/// download pointee data
	virtual auto data_impl() const -> result_or_err<sp_obj> = 0;

	/// download pointee structure -- link provide default implementation via `data_ex()` call
	virtual auto data_node_impl() const -> result_or_err<sp_node>;

	///////////////////////////////////////////////////////////////////////////////
	//  misc API for inherited links
	//

	// if pointee is a node - set node's handle to self and return pointee
	// default implementation obtains node via `data_node_ex()` and sets it's handle to self
	// but derived link can change default behaviour
	virtual auto propagate_handle() -> result_or_err<sp_node>;
	/// set handle of passed node to self
	auto self_handle_node(const sp_node& N) -> void;

	/// obtain inode pointer
	/// default impl do it via `data_ex()` call
	virtual auto get_inode() const -> result_or_err<inodeptr>;
	/// create or set or create inode for given target object
	/// [NOTE] if `new_info` is non-null, returned inode may be NOT EQUAL to `new_info`
	static auto make_inode(const sp_obj& target, inodeptr new_info = nullptr) -> inodeptr;

	// PIMPL
	struct impl;
	// some derived links may need to access base link's internals
	auto pimpl() const -> impl*;

private:
	std::unique_ptr<impl> pimpl_;
};
using sp_link = link::sp_link;
using sp_clink = link::sp_clink;

///////////////////////////////////////////////////////////////////////////////
//  link with bundled inode
//
class BS_API ilink : public link {
protected:
	/// ctor accept name of created link
	ilink(std::string name, const sp_obj& data, Flags f = Plain);

	/// return stored inode pointer
	auto get_inode() const -> result_or_err<inodeptr> override final;

private:
	inodeptr inode_;

	// serialization support
	friend class blue_sky::atomizer;
};

/*-----------------------------------------------------------------------------
 *  hard link stores direct pointer to object
 *  multiple hard links can point to the same object
 *-----------------------------------------------------------------------------*/
class BS_API hard_link : public ilink {
	friend class blue_sky::atomizer;

public:
	/// ctor -- additionaly accepts a pointer to object
	hard_link(std::string name, sp_obj data, Flags f = Plain);

	/// implement link's API
	auto clone(bool deep = false) const -> sp_link override;

	auto type_id() const -> std::string override;

protected:
	sp_obj data_;

private:
	auto data_impl() const -> result_or_err<sp_obj> override;

	//result_or_err<sp_node> data_node_impl() const override;
};

/*-----------------------------------------------------------------------------
 *  weak link is same as hard link, but stores weak link to data
 *  intended to be used to add class memebers self tree structure
 *-----------------------------------------------------------------------------*/
class BS_API weak_link : public ilink {
	friend class blue_sky::atomizer;

public:
	/// ctor -- additionaly accepts a pointer to object
	weak_link(std::string name, const sp_obj& data, Flags f = Plain);

	/// implement link's API
	auto clone(bool deep = false) const -> sp_link override;

	auto type_id() const -> std::string override;

private:
	std::weak_ptr<objbase> data_;

	auto data_impl() const -> result_or_err<sp_obj> override;
};

/*-----------------------------------------------------------------------------
 *  symbolic link is actually a link to another link, which is specified
 *  as absolute or relative string path
 *-----------------------------------------------------------------------------*/
class BS_API sym_link : public link {
	friend class blue_sky::atomizer;

public:
	/// ctor -- pointee is specified by string path
	sym_link(std::string name, std::string path, Flags f = Plain);
	/// ctor -- pointee is specified directly - absolute path will be stored
	sym_link(std::string name, const sp_clink& src, Flags f = Plain);

	/// implement link's API
	auto clone(bool deep = false) const -> sp_link override;

	auto type_id() const -> std::string override;

	/// additional sym link API
	/// check is pointed link is alive
	auto is_alive() const -> bool;

	/// return stored pointee path
	auto src_path(bool human_readable = false) const -> std::string;

private:
	std::string path_;

	auto reset_owner(const sp_node& new_owner) -> void override;

	auto data_impl() const -> result_or_err<sp_obj> override;

	auto propagate_handle() -> result_or_err<sp_node> override;
};


NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

