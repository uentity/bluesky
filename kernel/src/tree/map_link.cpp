/// @date 14.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/map_link.h>
#include <bs/meta.h>
#include <bs/log.h>

#include "map_engine.h"
#include "nil_engine.h"

#include <algorithm>

NAMESPACE_BEGIN(blue_sky::tree)
using ei = engine::impl;

map_link::map_link(
	uuid tag, std::string name, mapper_f mf, link_or_node src_node, link_or_node dest_node,
	Event update_on, TreeOpts opts, Flags f
) : super([&] {
		return visit(meta::overloaded{
			[&](link_mapper_f lmf) -> sp_engine_impl {
				return std::make_shared<map_link_impl>(
					tag, std::move(lmf), std::move(name), src_node, dest_node, update_on, opts, f
				);
			},
			[&](node_mapper_f nmf) -> sp_engine_impl {
				return std::make_shared<map_node_impl>(
					tag, std::move(nmf), std::move(name), src_node, dest_node, update_on, opts, f
				);
			}
		}, std::move(mf));
	}())
{}

map_link::map_link(
	std::string name, mapper_f mf, link_or_node src_node, link_or_node dest_node,
	Event update_on, TreeOpts opts, Flags f
) : map_link(
	gen_uuid(), std::move(name), std::move(mf), std::move(src_node), std::move(dest_node), update_on, opts, f
) {}


map_link::map_link(const map_link& rhs, mapper_f mf, link_or_node src_node, link_or_node dest_node) :
	map_link(
		ei::pimpl(rhs).tag_, ei::pimpl(rhs).name_, std::move(mf), std::move(src_node),
		dest_node.index() == 0 || std::get<1>(dest_node).is_nil() ?
			ei::pimpl(rhs).out_ : std::get<1>(dest_node),
		ei::pimpl(rhs).update_on_, ei::pimpl(rhs).opts_, ei::pimpl(rhs).flags_
	)
{}

map_link::map_link(const link& rhs) : super(rhs, type_id_()) {}

auto map_link::type_id_() -> std::string_view { return "map_link"; }

auto map_link::tag() const -> uuid {
	return ei::pimpl(*this).tag_;
}

auto map_link::input() const -> node {
	return ei::pimpl<map_link_impl_base>(*this).in_;
}

auto map_link::output() const -> node {
	return ei::pimpl<map_link_impl_base>(*this).out_;
}

auto map_link::l_target() const -> const link_mapper_f* {
	auto& simpl = ei::pimpl<map_link_impl_base>(*this);
	if(simpl.is_link_mapper)
		return &ei::pimpl<map_link_impl>(*this).mf_;
	return nullptr;
}

auto map_link::n_target() const -> const node_mapper_f* {
	auto& simpl = ei::pimpl<map_link_impl_base>(*this);
	if(!simpl.is_link_mapper)
		return &ei::pimpl<map_node_impl>(*this).mf_;
	return nullptr;
}

/*-----------------------------------------------------------------------------
 *  bundled mappers
 *-----------------------------------------------------------------------------*/
auto make_otid_filter(
	std::string name, std::vector<std::string> allowed_otids, link_or_node src_node, link_or_node dest_node,
	Event update_on, TreeOpts opts, Flags f
) -> map_link {
	std::sort(allowed_otids.begin(), allowed_otids.end());
	// [NOTE] dest link is ignored
	return map_link(
		std::move(name), [otids = std::move(allowed_otids)](link src, link /* dest */) -> link {
			if(std::binary_search(otids.begin(), otids.end(), src.obj_type_id()))
				return link(src.name(), src.data(), Flags::Plain);
			return link{};
		}, std::move(src_node), std::move(dest_node), update_on, opts, f
	);
}

NAMESPACE_END(blue_sky::tree)
