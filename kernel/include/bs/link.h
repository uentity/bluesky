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
#include "objbase.h"
#include <boost/uuid/uuid.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

/// base class of all links
class BS_API link  : public std::enable_shared_from_this<link> {
public:
	// give node access to all link's memebers
	friend class node;

	using id_type = boost::uuids::uuid;
	using sp_link = std::shared_ptr< link >;

	/// ctor accept name of created link
	link(std::string name);

	/// direct copying of links change ID
	link(const link&);

	/// virtual dtor
	virtual ~link();

	/// because we cannot make explicit copies of link
	/// we need a dedicated faunction to make links clones
	virtual sp_link clone() const = 0;

	/// get pointer to object link is pointing to
	/// NOTE: returned pointer can be null
	virtual sp_obj data() const = 0;

	/// query what kind of link is this
	virtual std::string type_id() const = 0;

	/// get link's object ID
	virtual std::string oid() const;

	/// get link's object type ID
	virtual std::string obj_type_id() const;

	/// return tree::node if contained object is a node
	/// derived class can return cached node info
	virtual sp_node data_node() const = 0;

	/// get/set object's inode
	virtual inode info() const = 0;
	virtual void set_info(inodeptr i) = 0;

	/// flags reflect link properties and state
	enum Flags {
		Persistent = 1,
		Disabled = 2
	};
	virtual uint flags() const;
	virtual void set_flags(uint new_flags);

	/// access link's unique ID
	const id_type& id() const {
		return id_;
	}

	/// obtain link's symbolic name
	std::string name() const {
		return name_;
	}

	/// get link's container
	sp_node owner() const {
		return owner_.lock();
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

protected:
	std::string name_;
	id_type id_;
	uint flags_;
	std::weak_ptr<node> owner_;

	friend class node;
	void reset_owner(const sp_node& new_owner);
};

/// hard link stores direct pointer to object
/// there can exist many hard links to single object
class BS_API hard_link : public link {
public:

	/// ctor -- additionaly accepts a pointer to object
	hard_link(std::string name, sp_obj data);

	/// implement link's API
	sp_link clone() const override;

	sp_obj data() const override;

	std::string type_id() const override;

	std::string oid() const override;

	std::string obj_type_id() const override;

	sp_node data_node() const override;

	inode info() const override;
	void set_info(inodeptr i) override;

private:
	sp_obj data_;
};

/// weak link is same as hard link, but stores weak link to data
/// intended to be used to add class memebers self tree structure
class BS_API weak_link : public link {
public:

	/// ctor -- additionaly accepts a pointer to object
	weak_link(std::string name, const sp_obj& data);

	/// implement link's API
	sp_link clone() const override;

	sp_obj data() const override;

	std::string type_id() const override;

	std::string oid() const override;

	std::string obj_type_id() const override;

	sp_node data_node() const override;

	inode info() const override;
	void set_info(inodeptr i) override;

private:
	std::weak_ptr<objbase> data_;
};


NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

