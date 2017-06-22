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

/// base class of all links
class BS_API bs_link {
public:
	// give node access to all link's memebers
	friend class bs_node;

	using id_type = boost::uuids::uuid;
	using sp_link = std::shared_ptr< bs_link >;

	enum link_type {
		hard_link = 0,
		sym_link,
		net_link
	};

	/// ctor accept name of created link
	bs_link(std::string name);

	/// direct copying of links are prohibited
	bs_link(const bs_link&);

	/// virtual dtor
	virtual ~bs_link();

	/// because we cannot make explicit copies of link
	/// we need a dedicated faunction to make links clones
	virtual sp_link clone() const = 0;

	/// access link's unique ID
	const id_type& id() const {
		return id_;
	}

	/// obtain link's symbolic name
	std::string name() const {
		return name_;
	}

	/// get pointer to object link is pointing to
	/// NOTE: returned pointer can be null
	virtual sp_obj data() const = 0;

	/// query what kind of link is this
	virtual link_type type_id() const = 0;

	/// get link's parent container
	//virtual sp_link parent() const = 0;

protected:
	std::string name_;
	id_type id_;
};

/// hard link stores direct pointer to object
/// there can exist many hard links to single object
class BS_API bs_hard_link : public bs_link {
public:

	/// ctor -- additionaly accepts a pointer to object
	bs_hard_link(std::string name, const sp_obj& data);

	/// implement link's API
	sp_obj data() const override;

	link_type type_id() const override;

	sp_link clone() const override;

	//std::shared_ptr< bs_link > parent() const override;

private:
	sp_obj data_;
};

NAMESPACE_END(blue_sky)

