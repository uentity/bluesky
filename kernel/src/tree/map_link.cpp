/// @date 14.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/map_link.h>
#include <bs/meta.h>
#include <bs/log.h>

#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

#include "map_engine.h"
#include "nil_engine.h"

#include <algorithm>

NAMESPACE_BEGIN(blue_sky::tree)
using ei = engine::impl;

map_link::map_link(
	mapper_f mf, uuid tag, std::string name, link_or_node src_node, link_or_node dest_node,
	Event update_on, TreeOpts opts, Flags f
) : super([&] {
		const auto make_lmapper_impl = [&](link_mapper_f lmf) -> sp_engine_impl {
			return std::make_shared<map_link_impl>(
				std::move(lmf), tag, std::move(name), src_node, dest_node, update_on, opts, f
			);
		};
		const auto make_nmapper_impl = [&](node_mapper_f nmf) -> sp_engine_impl {
			return std::make_shared<map_node_impl>(
				std::move(nmf), tag, std::move(name), src_node, dest_node, update_on, opts, f
			);
		};

		return visit(meta::overloaded{

			[&](link_mapper_f lmf) -> sp_engine_impl { return make_lmapper_impl(std::move(lmf)); },

			[&](simple_link_mapper_f lmf) -> sp_engine_impl {
				return make_lmapper_impl([lmf = std::move(lmf)](link src, link dest, caf::event_based_actor*) {
					return caf::result<link>{ lmf(src, dest) };
				});
			},

			[&](node_mapper_f nmf) -> sp_engine_impl { return make_nmapper_impl(std::move(nmf)); },

			[&](simple_node_mapper_f nmf) -> sp_engine_impl {
				return make_nmapper_impl([nmf = std::move(nmf)](node src, node dest, caf::event_based_actor*) {
					nmf(src, dest);
					return caf::result<void>{};
				});
			},

		}, std::move(mf));
	}())
{}

map_link::map_link(
	mapper_f mf, std::string name, link_or_node src_node, link_or_node dest_node,
	Event update_on, TreeOpts opts, Flags f
) : map_link(
	std::move(mf), gen_uuid(), std::move(name), std::move(src_node), std::move(dest_node), update_on, opts, f
) {}


map_link::map_link(mapper_f mf, const map_link& rhs, link_or_node src_node, link_or_node dest_node) :
	map_link(
		std::move(mf), ei::pimpl(rhs).tag_, ei::pimpl(rhs).name_, std::move(src_node),
		dest_node.index() == 0 || std::get<1>(dest_node).is_nil() ?
			ei::pimpl(rhs).out_ : std::get<1>(dest_node),
		ei::pimpl(rhs).update_on_, ei::pimpl(rhs).opts_, ei::pimpl(rhs).flags_
	)
{
	// ensure that dest node is relinked to this new link
	pimpl()->propagate_handle(ei::pimpl(*this).out_);
	rs_reset(Req::DataNode, rhs.req_status(Req::DataNode));
}

map_link::map_link(const link& rhs) : super(rhs, type_id_()) {}

auto map_link::type_id_() -> std::string_view { return "map_link"; }

auto map_link::tag() const -> uuid {
	return ei::pimpl(*this).tag_;
}

auto map_link::input() const -> node {
	return ei::pimpl<map_impl_base>(*this).in_;
}

auto map_link::output() const -> node {
	return ei::pimpl<map_impl_base>(*this).out_;
}

auto map_link::l_target() const -> const link_mapper_f* {
	auto& simpl = ei::pimpl<map_impl_base>(*this);
	if(simpl.is_link_mapper)
		return &ei::pimpl<map_link_impl>(*this).mf_;
	return nullptr;
}

auto map_link::n_target() const -> const node_mapper_f* {
	auto& simpl = ei::pimpl<map_impl_base>(*this);
	if(!simpl.is_link_mapper)
		return &ei::pimpl<map_node_impl>(*this).mf_;
	return nullptr;
}

/*-----------------------------------------------------------------------------
 *  bundled mappers
 *-----------------------------------------------------------------------------*/
auto make_otid_filter(
	std::vector<std::string> allowed_otids, std::string name, link_or_node src_node, link_or_node dest_node,
	Event update_on, TreeOpts opts, Flags f
) -> map_link {
	std::sort(allowed_otids.begin(), allowed_otids.end());
	// [NOTE] dest link is ignored
	return map_link(
		[otids = std::move(allowed_otids)](link src, link /* dest */, caf::event_based_actor* worker)
		-> caf::result<link> {
			auto res = worker->make_response_promise<link>();
			auto src_actor = src.actor();

			worker->request(src_actor, kernel::radio::timeout(), a_lnk_otid{})
			.then([=](const std::string& otid) mutable {
				if(std::binary_search(otids.begin(), otids.end(), otid))
					res.delegate(src_actor, a_clone{}, false);
				else
					res.deliver(link{});
			});
			return link{};
		}, std::move(name), std::move(src_node), std::move(dest_node), update_on, opts, f
	);
}

NAMESPACE_END(blue_sky::tree)
