/// @date 14.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/map_link.h>
#include <bs/meta.h>

#include "map_engine.h"
#include "nil_engine.h"

NAMESPACE_BEGIN(blue_sky::tree)

map_link::map_link(
	std::string name, mapper_f mf, link_or_node src_node, link_or_node dest_node,
	Event update_on, TreeOpts opts, Flags f
) : super([&] {
		return visit(meta::overloaded{
			[&](link_mapper_f lmf) -> sp_engine_impl {
				return std::make_shared<map_link_impl>(
					std::move(name), std::move(lmf), src_node, dest_node, update_on, opts, f
				);
			},
			// [NOTE] not implemented yet
			[&](const node_mapper_f& nmf) -> sp_engine_impl { return nil_link::pimpl(); }
		}, std::move(mf));
	}())
{}

map_link::map_link(const link& rhs) : super(rhs, type_id_()) {}

auto map_link::type_id_() -> std::string_view { return "map_link"; }

auto map_link::l_target() const -> const link_mapper_f* {
	return &static_cast<map_link_impl const*>(pimpl())->mf_;
}

auto map_link::n_target() const -> const node_mapper_f* {
	return nullptr;
}

NAMESPACE_END(blue_sky::tree)
