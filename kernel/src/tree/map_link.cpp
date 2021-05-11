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
#include <memory_resource>

NAMESPACE_BEGIN(blue_sky::tree)
using ei = engine::impl;

// setup synchronized pool allocator for link impls
static auto impl_pool = std::pmr::synchronized_pool_resource{};
static auto map_link_alloc = std::pmr::polymorphic_allocator<map_link_impl>(&impl_pool);
static auto map_node_alloc = std::pmr::polymorphic_allocator<map_node_impl>(&impl_pool);

map_link::map_link(
	mapper_f mf, uuid tag, std::string name, link_or_node src_node, link_or_node dest_node,
	Event update_on, TreeOpts opts, Flags f
) : super([&] {
		const auto make_lmapper_impl = [&](link_mapper_f lmf) -> sp_engine_impl {
			return std::allocate_shared<map_link_impl>(
				map_link_alloc, std::move(lmf), tag, std::move(name), src_node, dest_node, update_on, opts, f
			);
		};
		const auto make_nmapper_impl = [&](node_mapper_f nmf) -> sp_engine_impl {
			return std::allocate_shared<map_node_impl>(
				map_node_alloc, std::move(nmf), tag, std::move(name), src_node, dest_node, update_on, opts, f
			);
		};

		return visit(meta::overloaded{

			[&](link_mapper_f lmf) -> sp_engine_impl { return make_lmapper_impl(std::move(lmf)); },

			[&](simple_link_mapper_f lmf) -> sp_engine_impl {
				return make_lmapper_impl([lmf = std::move(lmf)](auto src, auto dest, auto ev, auto*) {
					return caf::result<link>{ lmf(src, dest, std::move(ev)) };
				});
			},

			[&](node_mapper_f nmf) -> sp_engine_impl { return make_nmapper_impl(std::move(nmf)); },

			[&](simple_node_mapper_f nmf) -> sp_engine_impl {
				return make_nmapper_impl([nmf = std::move(nmf)](auto src, auto dest, auto ev, auto*) {
					nmf(src, dest, std::move(ev));
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


static auto is_empty(const link_or_node& x) {
	return std::visit([](const auto& v) { return v.is_nil(); }, x);
}

// construct from existing link + install mapping function
map_link::map_link(
	mapper_f mf, const map_link& rhs,
	link_or_node src_node, link_or_node dest_node, TreeOpts opts
) :
	map_link(
		std::move(mf), ei::pimpl(rhs).tag_, ei::pimpl(rhs).name_, std::move(src_node),
		[&]() -> link_or_node {
			if(is_empty(dest_node)) {
				auto& rhs_out = ei::pimpl(rhs).out_;
				if(rhs_out.handle() == rhs)
					// if `rhs` owns it's output node, rebind it to tmp link first
					return link::make_root(std::string{}, ei::pimpl(rhs).out_);
				else
					return rhs_out;
			}
			else
				return std::move(dest_node);
		}(),
		ei::pimpl(rhs).update_on_, enumval(opts) ? opts : ei::pimpl(rhs).opts_, ei::pimpl(rhs).flags_
	)
{
	// copy DataNode status from rhs
	const auto& self_out = ei::pimpl(*this).out_;
	if(self_out && self_out == ei::pimpl(rhs).out_)
		pimpl()->rs_reset_quiet(Req::DataNode, ReqReset::Always, rhs.req_status(Req::DataNode));
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
	namespace kradio = kernel::radio;
	std::sort(allowed_otids.begin(), allowed_otids.end());
	// [NOTE] dest link is ignored
	return map_link(
		[otids = std::move(allowed_otids)]
		(link src, link /* dest */, event /* ev */, caf::event_based_actor* worker) -> caf::result<link> {
			auto res = worker->make_response_promise<link>();
			auto src_actor = src.actor();

			worker->set_error_handler([=](auto* worker, auto& er) mutable {
				res.deliver(link{});
				worker->default_error_handler(worker, er);
			});

			worker->request(src_actor, kradio::timeout(true), a_data{}, true)
			.then([=](obj_or_errbox maybe_obj) mutable {
				if(maybe_obj && std::binary_search(otids.begin(), otids.end(), (*maybe_obj)->type_id()))
					worker->request(src_actor, kradio::timeout(), a_lnk_name{})
					.then([=, obj = *std::move(maybe_obj)](std::string src_name) mutable {
						res.deliver(link{ weak_link(std::move(src_name), obj) });
					});
				else
					res.deliver(link{});
			});
			return res;
		}, std::move(name), std::move(src_node), std::move(dest_node), update_on, opts, f
	);
}

NAMESPACE_END(blue_sky::tree)
