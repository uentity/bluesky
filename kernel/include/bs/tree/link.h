/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief BS tree link class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once
#include "../objbase.h"
#include "../detail/enumops.h"
#include "../error.h"

#include <atomic>
#include <chrono>
#include <boost/uuid/uuid.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

// time point type for all timestaps
using time_point = std::chrono::system_clock::time_point;

/// inode that stores access rights, timestampts, etc
struct BS_API inode {
	// flags
	bool : 1;
	bool suid : 1;
	bool sgid : 1;
	bool sticky : 1;

	// access rights
	// user (owner)
	bool : 1;
	std::uint8_t u : 3;
	// group
	bool : 1;
	std::uint8_t g : 3;
	// others
	bool : 1;
	std::uint8_t o : 3;

	// modification time
	time_point mod_time;
	// link's owner
	std::string owner;
	std::string group;

	// do std initialization of all values
	inode();
};
using inodeptr = std::unique_ptr<inode>;

/// base class of all links
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
	const id_type& id() const;

	/// obtain link's symbolic name
	std::string name() const;

	/// get link's container
	sp_node owner() const;

	/// get/set object's inode
	auto info() const -> inode;
	auto set_info(inode I) -> void;

	/// flags reflect link properties and state
	enum Flags {
		Plain = 0,
		Persistent = 1,
		Disabled = 2
	};
	Flags flags() const;
	void set_flags(Flags new_flags);

	/// rename link & notify owner node
	auto rename(std::string new_name) -> void;

	/// because we cannot make explicit copies of link
	/// we need a dedicated function to make links clones
	/// if `deep` flag is set, then clone pointed object as well
	virtual sp_link clone(bool deep = false) const = 0;

	/// query what kind of link is this
	virtual std::string type_id() const = 0;

	///////////////////////////////////////////////////////////////////////////////
	//  sync API
	//
	/// get link's object ID -- fast, can return empty string
	virtual std::string oid() const;

	/// get link's object type ID -- fast, can return nil type ID
	virtual std::string obj_type_id() const;

	/// get pointer to object link is pointing to -- slow, never returns invalid (NULL) sp_obj
	/// NOTE: returned pointer can be null
	result_or_err<sp_obj> data_ex() const;
	/// simple data accessor that returns nullptr on error
	sp_obj data() const {
		return data_ex().value_or(nullptr);
	}

	/// return tree::node if contained object is a node -- slow, never returns invalid (NULL) sp_obj
	/// derived class can return cached node info
	result_or_err<sp_node> data_node_ex() const;
	/// simple tree::node accessor that returns nullptr on error
	sp_node data_node() const {
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
	ReqStatus req_status(Req request) const;
	// unconditional reset request status
	ReqStatus rs_reset(Req request, ReqStatus new_status = ReqStatus::Void);
	ReqStatus rs_reset_if_eq(Req request , ReqStatus self_rs, ReqStatus new_rs = ReqStatus::Void);
	ReqStatus rs_reset_if_neq(Req request, ReqStatus self_rs, ReqStatus new_rs = ReqStatus::Void);

	/// obtain data in async manner passing it to callback
	using process_data_cb = std::function<void(result_or_err<sp_obj>, sp_clink)>;
	auto data(process_data_cb f) const -> void;
	/// ... and data node
	auto data_node(process_data_cb f) const -> void;

protected:
	// serialization support
	friend class blue_sky::atomizer;
	// full access for node
	friend class node;

	/// ctor accept name of created link
	link(std::string name, Flags f = Plain);

	/// direct copying of links change ID
	link(const link&) = delete;

	// silent replace old name with new in link's internals
	auto rename_silent(std::string new_name) -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  sync API
	//  Implementation need not to bother with reuqest status
	//  Performance hint: you can check if status is OK
	//  to omit useless repeating possibly long operation calls
	//

	/// switch link's owner
	virtual void reset_owner(sp_node new_owner);

	/// download pointee data
	virtual result_or_err<sp_obj> data_impl() const = 0;

	/// download pointee structure -- link provide default implementation via `data_ex()` call
	virtual result_or_err<sp_node> data_node_impl() const;

	using method = std::function<result_or_err<sp_obj> (link*)>;
	using const_method = std::function<result_or_err<sp_obj> (const link*)>;

	/// helper to invoke any given link method and atomically set status
	auto invoke(
		method f, ReqStatus& status, std::atomic_flag& status_flag
	) -> result_or_err<sp_obj>;

	auto invoke(
		const_method f, ReqStatus& status, std::atomic_flag& status_flag
	) const -> result_or_err<sp_obj>;

	// PIMPL
	struct impl;
	std::unique_ptr<impl> pimpl_;
};
using sp_link = link::sp_link;
using sp_clink = link::sp_clink;

/// hard link stores direct pointer to object
/// there can exist many hard links to single object
class BS_API hard_link : public link {
	friend class blue_sky::atomizer;

public:
	/// ctor -- additionaly accepts a pointer to object
	hard_link(std::string name, sp_obj data, Flags f = Plain);

	/// implement link's API
	sp_link clone(bool deep = false) const override;

	std::string type_id() const override;

protected:
	sp_obj data_;

private:
	result_or_err<sp_obj> data_impl() const override;

	//result_or_err<sp_node> data_node_impl() const override;
};

/// weak link is same as hard link, but stores weak link to data
/// intended to be used to add class memebers self tree structure
class BS_API weak_link : public link {
	friend class blue_sky::atomizer;

public:
	/// ctor -- additionaly accepts a pointer to object
	weak_link(std::string name, const sp_obj& data, Flags f = Plain);

	/// implement link's API
	sp_link clone(bool deep = false) const override;

	std::string type_id() const override;

private:
	std::weak_ptr<objbase> data_;

	result_or_err<sp_obj> data_impl() const override;
};

/// symbolic link is actually a link to another link, which is specified as absolute or relative
/// string path
class BS_API sym_link : public link {
	friend class blue_sky::atomizer;
public:

	/// ctor -- pointee is specified by string path
	sym_link(std::string name, std::string path, Flags f = Plain);
	/// ctor -- pointee is specified directly - absolute path will be stored
	sym_link(std::string name, const sp_clink& src, Flags f = Plain);

	/// implement link's API
	sp_link clone(bool deep = false) const override;

	std::string type_id() const override;

	/// additional sym link API
	/// check is pointed link is alive
	bool is_alive() const;

	/// return stored pointee path
	std::string src_path(bool human_readable = false) const;

private:
	std::string path_;

	void reset_owner(sp_node new_owner) override;

	result_or_err<sp_obj> data_impl() const override;
};


NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

