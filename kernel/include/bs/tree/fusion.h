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

NAMESPACE_BEGIN(blue_sky::tree)

/*-----------------------------------------------------------------------------
 *  Interface of Fusion brigde that implements filling target item
 *-----------------------------------------------------------------------------*/
class BS_API fusion_iface {
public:
	/// download object's 'Data' content from third-party backend
	auto pull_data(sp_obj root, link root_link, prop::propdict params = {}) -> error;
	/// download object's children metadata (fill bundled node)
	auto populate(sp_obj root, link root_link, prop::propdict params = {}) -> error;
	// [NOTE] `pull_data()` and `populate()` can be invoked in parallel

	/// test if object is either pure container or pure data and can be fetched using single request
	/// for such objects Data & DataNode statuses will be changed simultaneousely
	/// [NOTE] default imlementation always returns `true`
	/// (most objects are uniform, so match common case)
	virtual auto is_uniform(const sp_obj& root) const -> bool;

	virtual ~fusion_iface() = default;

private:
	/// derived fusion bridges must override these
	virtual auto do_pull_data(sp_obj root, link root_link, prop::propdict params) -> error = 0;
	virtual auto do_populate(sp_obj root, link root_link, prop::propdict params) -> error = 0;
};
using sp_fusion = std::shared_ptr<fusion_iface>;

/*-----------------------------------------------------------------------------
 *  Fusion link populates object children when `data_node()` or `data()` is called
 *-----------------------------------------------------------------------------*/
struct fusion_link_impl;

class BS_API fusion_link : public link {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = link;
	using engine_impl = fusion_link_impl;

	using fusion_actor_type = caf::typed_actor<
		// populate pointee with given children types
		caf::replies_to<a_flnk_data, prop::propdict, bool>::with<obj_or_errbox>,
		// populate pointee with given children types
		caf::replies_to<a_flnk_populate, prop::propdict, bool>::with<node_or_errbox>,
		// get link's bridge
		caf::replies_to<a_flnk_bridge>::with<sp_fusion>,
		// set link's bridge
		caf::reacts_to<a_flnk_bridge, sp_fusion>
	>;

	using actor_type = link::actor_type::extend_with<fusion_actor_type>;

	// ctors
	fusion_link(
		std::string name, sp_obj data = nullptr,
		sp_fusion bridge = nullptr, Flags f = Plain
	);
	fusion_link(
		std::string name, node folder,
		sp_fusion bridge = nullptr, Flags f = Plain
	);

	fusion_link(
		std::string name, const char* obj_type, std::string oid = "",
		sp_fusion bridge = nullptr, Flags f = Plain
	);
	/// convert from base link
	fusion_link(const link& rhs);

	static auto type_id_() -> std::string_view;

	// provide access to Fusion API of installed bridge
	auto pull_data(prop::propdict params = {}, bool wait_if_busy = true) const -> obj_or_err;
	auto pull_data(process_data_cb f, prop::propdict params) const -> void;

	auto populate(prop::propdict params = {}, bool wait_if_busy = true) const -> node_or_err;
	auto populate(process_dnode_cb f, prop::propdict params) const -> void;

	// access to link's fusion bridge
	auto bridge() const -> sp_fusion;
	auto reset_bridge(sp_fusion new_bridge = nullptr) -> void;

private:
	// don't start internal actor
	fusion_link();
};

NAMESPACE_END(blue_sky::tree)

// support for hashed container of links
STD_HASH_BS_LINK(fusion_link)
