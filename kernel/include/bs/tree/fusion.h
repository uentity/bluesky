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

	virtual ~fusion_iface();
};
using sp_fusion = std::shared_ptr<fusion_iface>;

/*-----------------------------------------------------------------------------
 *  Fusion link populates object children when `data_node()` or `data()` is called
 *-----------------------------------------------------------------------------*/
class BS_API fusion_link : public ilink {
	friend class blue_sky::atomizer;

public:
	// ctors
	fusion_link(
		std::string name, sp_node data = nullptr,
		sp_fusion bridge = nullptr, Flags f = Plain
	);
	fusion_link(
		std::string name, const char* obj_type, std::string oid = "",
		sp_fusion bridge = nullptr, Flags f = Plain
	);
	// dtor
	~fusion_link();

	auto clone(bool deep = false) const -> sp_link override;

	// link API implementation
	auto type_id() const -> std::string override;
	//auto oid() const -> std::string override;
	//auto obj_type_id() const -> std::string override;

	// force `fusion_iface::populate()` call with specified children types
	// regardless of populate status
	auto populate(const std::string& child_type_id, bool wait_if_busy = false) const
		-> result_or_err<sp_node>;
	// async populate
	auto populate(process_data_cb f, std::string child_type_id, bool wait_if_busy = true) const
		-> void;

	// access to link's fusion bridge
	auto bridge() const -> sp_fusion;
	auto reset_bridge(sp_fusion new_bridge = nullptr) -> void;

private:
	struct impl;
	std::unique_ptr<impl> pimpl_;

	// pull object's data via `fusion_iface::pull_data()`
	auto data_impl() const -> result_or_err<sp_obj> override;
	// pull leafs via `fusion_iface::populate()`
	auto data_node_impl() const -> result_or_err<sp_node> override;
	// bypass `data_node_ex()` call and directly set handle of contained node object
	auto propagate_handle() -> result_or_err<sp_node> override;
};
using sp_fusion_link = std::shared_ptr<fusion_link>;
using sp_cfusion_link = std::shared_ptr<const fusion_link>;

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

