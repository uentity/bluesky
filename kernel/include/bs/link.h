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

/// inode that stores access rights, timestampts, etc
struct BS_API inode {
	// link's owner
	std::string owner;
	std::string group;

	// flags
	bool : 1;
	bool suid : 1;
	bool sgid : 1;
	bool sticky : 1;

	// access rights
	// user (owner)
	bool : 1;
	unsigned int u : 3;
	// group
	bool : 1;
	unsigned int g : 3;
	// others
	bool : 1;
	unsigned int o : 3;
};

/// base class of all links
class BS_API link {
public:
	// give node access to all link's memebers
	friend class node;

	using id_type = boost::uuids::uuid;
	using sp_link = std::shared_ptr< link >;

	enum LinkType {
		Hard = 0,
		Symbolic,
		Net
	};

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
	virtual LinkType type_id() const = 0;

	/// get link's object ID
	virtual std::string oid() const;

	/// get link's object type ID
	virtual std::string obj_type_id() const;

	/// access link's unique ID
	const id_type& id() const {
		return id_;
	}

	/// obtain link's symbolic name
	std::string name() const {
		return name_;
	}

	const inode& get_inode() const {
		return inode_;
	}
	inode& get_inode() {
		return inode_;
	}

	/// get link's container
	sp_node owner() const {
		return owner_.lock();
	}

protected:
	std::string name_;
	id_type id_;
	inode inode_;
	std::weak_ptr<node> owner_;

	friend class node;
	void reset_owner(const sp_node& new_owner);
};

/// hard link stores direct pointer to object
/// there can exist many hard links to single object
class BS_API hard_link : public link {
public:

	/// ctor -- additionaly accepts a pointer to object
	hard_link(std::string name, const sp_obj& data);

	/// implement link's API
	sp_obj data() const override;

	LinkType type_id() const override;

	sp_link clone() const override;

private:
	sp_obj data_;
};

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

