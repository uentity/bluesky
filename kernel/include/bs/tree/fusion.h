/// @file
/// @author uentity
/// @date 09.08.2018
/// @brief Fusion link and Fusion client interface declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "link.h"

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  Interface of Fusion client that have to populate root item with children
 *-----------------------------------------------------------------------------*/
class BS_API fusion_iface {
public:
	/// accept root object and optionally type of child objects to be populated with
	virtual auto populate(const sp_node& root, const std::string& child_type_id = "") -> error = 0;
	/// download passed object's content from third-party backend
	virtual auto pull_data(const sp_obj& root) -> error = 0;
};
using sp_fusion = std::shared_ptr<fusion_iface>;

/*-----------------------------------------------------------------------------
 *  Fusion link populates object children when `data_node()` or `data()` is called
 *-----------------------------------------------------------------------------*/
class BS_API fusion_link : public link {
	friend class blue_sky::atomizer;
public:

	// ctors
	fusion_link(
		sp_fusion bridge, std::string name, sp_node data, Flags f = Persistent
	);
	fusion_link(
		sp_fusion bridge, std::string name, const char* obj_type,
		std::string oid = "", Flags f = Persistent
	);
	// dtor
	~fusion_link();

	auto clone(bool deep = false) const -> sp_link override;

	// link API implementation
	auto type_id() const -> std::string override;
	auto oid() const -> std::string override;
	auto obj_type_id() const -> std::string override;

	// pull object's data via `fusion_iface::pull_data()`
	auto data() const -> sp_obj override;
	// pull leafs via `fusion_iface::populate()`
	auto data_node() const -> sp_node override;
	// force `fusion_iface::populate()` call with specified children types
	// regardless of populate status
	auto populate(const std::string& child_type_id) -> error;

	// enum states of request to fusion_iface
	enum class OpStatus { Void, Busy, OK, Error, Pending };

	// get/set populate status
	OpStatus populate_status() const;
	void reset_populate_status(OpStatus new_status = OpStatus::Void);

	// get/set pull data status
	OpStatus data_status() const;
	void reset_data_status(OpStatus new_status = OpStatus::Void);

private:
	struct fusion_link_impl;
	std::unique_ptr<fusion_link_impl> pimpl_;
};

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

