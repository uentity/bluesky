/// @file
/// @author uentity
/// @date 26.02.2020
/// @brief Qt model helper
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "link.h"
#include "node.h"

#include <cstdint>
#include <optional>

NAMESPACE_BEGIN(blue_sky::tree)

/// Context is designed to ease navigating BS tree, particularly follow different child <-> parent paths
/// Primary use case is as Qt model helper that generates internal pointers (item_tag ptr) for indexes
class BS_API context {
public:
	// pairs { item path -> item link } kept in a map
	using item_tag = std::pair<const lids_v, link::weak_ptr>;
	using existing_tag = std::optional<item_tag const*>;
	// combines above with link offset in parent's node
	using item_index = std::pair<existing_tag, std::int64_t>;

	context(node root = node::nil());
	context(sp_obj root);
	context(link root);
	// for unique_ptr
	~context();

	/// reset context to new root
	auto reset(link root) -> void;
	auto reset(node root, link root_handle = {}) -> void;

	/// simple accessors to model's data
	auto root() const -> node;
	auto root_link() const -> link;
	auto root_path(Key path_unit) const -> std::string;

	/// make tag for given path
	auto operator()(const std::string& path, bool nonexact_match = false) -> item_index;
	/// for given link + possible hint
	auto operator()(const link& L, std::string path_hint = "/") -> item_index;

	/// make child tag from parent and index of child in parent
	auto operator()(std::int64_t item_idx, existing_tag parent = {}) -> existing_tag;
	/// make parent tag from child one
	auto operator()(existing_tag child) -> item_index;

	/// DEBUG
	auto dump() const -> void;

private:
	struct impl;
	std::unique_ptr<impl> pimpl_;
};

NAMESPACE_END(blue_sky::tree)
