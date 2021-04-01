/// @author Alexander Gagarin (@uentity)
/// @date 17.02.2021
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "map_engine.h"
#include "request_impl.h"

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
NAMESPACE_BEGIN()

template<bool DiscardResult = false>
auto spawn_mapper_job(map_node_impl* mama, map_link_actor* papa, event ev)
-> std::conditional_t<DiscardResult, void, caf::result<node_or_errbox>> {
	// safely invoke mapper and return output node on success
	auto invoke_mapper =
		[mama = papa->spimpl<map_node_impl>(), mf = mama->mf_, ev = std::move(ev)]
		(caf::event_based_actor* worker) mutable -> caf::result<node_or_errbox> {
			auto invoke_res = worker->make_response_promise<node_or_errbox>();

			using res_t = caf::result<void>;
			worker->become(caf::message_handler{
				[=, mf = std::move(mf), ev = std::move(ev)](a_mlnk_fresh) mutable -> res_t {
					auto res = std::optional<res_t>{};
					if(auto er = error::eval_safe([&] {
						res = mf(mama->in_, mama->out_, std::move(ev), worker);
					})) {
						invoke_res.deliver(node_or_errbox{tl::unexpect, er.pack()});
					}
					return res ? std::move(*res) : caf::error();
				}
			}.or_else(worker->current_behavior()));

			worker->request(caf::actor_cast<caf::actor>(worker), caf::infinite, a_mlnk_fresh{})
			.then([=]() mutable {
				invoke_res.deliver(node_or_errbox{mama->out_});
			});
			return invoke_res;
		};

	auto opts = enumval(mama->opts_ & TreeOpts::DetachedWorkers) ?
		ReqOpts::Detached : ReqOpts::WaitIfBusy;
	if(enumval(mama->opts_ & TreeOpts::TrackWorkers))
		opts |= ReqOpts::TrackWorkers;

	if constexpr(DiscardResult) {
		// trigger async request by sending `a_ack` message to worker actor
		request_impl<node>(*papa, Req::DataNode, opts, std::move(invoke_mapper))
		.map([&](auto&& rworker) {
			papa->send(rworker, a_ack());
		});
	}
	else
		return request_data_impl<node>(*papa, Req::DataNode, opts, std::move(invoke_mapper));
}

NAMESPACE_END()

// default ctor installs noop mapping fn
map_node_impl::map_node_impl() :
	map_impl_base(false), mf_(noop_r<caf::result<void>>())
{}

auto map_node_impl::clone(link_actor*, bool deep) const -> caf::result<sp_limpl> {
	// [NOTE] output node is always brand new, otherwise a lot of questions & issues rises
	return std::make_shared<map_node_impl>(mf_, tag_, name_, in_, node::nil(), update_on_, opts_, flags_);
}

auto map_node_impl::erase(map_link_actor* papa, lid_type, event ev) -> void {
	spawn_mapper_job<true>(this, papa, std::move(ev));
}

auto map_node_impl::update(map_link_actor* papa, link, event ev) -> void {
	spawn_mapper_job<true>(this, papa, std::move(ev));
}

auto map_node_impl::refresh(map_link_actor* papa) -> caf::result<node_or_errbox> {
	return spawn_mapper_job(this, papa, event{caf::actor_cast<caf::actor>(papa), {}, Event::None});
}

// [NOTE] both link -> link & node -> node impls have same type ID,
// because it must match with `map_link::type_id_()`
ENGINE_TYPE_DEF(map_node_impl, "map_link")

NAMESPACE_END(blue_sky::tree)
