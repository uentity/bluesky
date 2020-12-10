/// @date 01.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "link.h"
#include "node.h"

#include <functional>
#include <variant>

NAMESPACE_BEGIN(blue_sky::tree)
using link_or_node = std::variant<link, node>;

class BS_API map_link : public link {
public:
	using super = link;
	using engine_impl = class map_link_impl_base;
	using link_mapper_f = std::function< link(link /* source */, link /* existing dest */) >;
	using node_mapper_f = std::function< void(node /* source */, node /* existing dest */) >;

	using mapper_f = std::variant<link_mapper_f, node_mapper_f>;

	/// can pass `link` or `node` as source and destination
	map_link(
		std::string name, mapper_f mf, link_or_node src_node, link_or_node dest_node = {},
		Event update_on = Event::DataModified, TreeOpts opts = TreeOpts::Normal, Flags f = Flags::Plain
	);
	/// with custom tag
	map_link(
		uuid tag, std::string name, mapper_f mf, link_or_node src_node, link_or_node dest_node = {},
		Event update_on = Event::DataModified, TreeOpts opts = TreeOpts::Normal, Flags f = Flags::Plain
	);
	/// construct from existing copy of map_link but with another mapping
	map_link(const map_link& rhs, mapper_f mf, link_or_node src_node, link_or_node dest_node = {});

	map_link(const link& rhs);

	static auto type_id_() -> std::string_view;

	auto tag() const -> uuid;
	auto input() const -> node;
	auto output() const -> node;

	auto l_target() const -> const link_mapper_f*;
	auto n_target() const -> const node_mapper_f*;
};

/// returns map_link that filters objects from input node by object type ID(s)
BS_API auto make_otid_filter(
	std::string name, std::vector<std::string> allowed_otids, link_or_node src_node, link_or_node dest_node = {},
	Event update_on = Event::DataNodeModified, TreeOpts opts = TreeOpts::Deep, Flags f = Flags::Plain
) -> map_link;

NAMESPACE_END(blue_sky::tree)
