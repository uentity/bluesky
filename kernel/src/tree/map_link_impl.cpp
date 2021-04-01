/// @date 05.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "map_engine.h"
#include "request_impl.h"

#include <bs/tree/tree.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
NAMESPACE_BEGIN()

using lmapper_res_t = std::pair< link /* mapping result */, link /* dest */ >;

// modify worker actor behavior s.t. it invokes link mapper on `a_ack` message
template<typename F>
auto make_lmapper_actor(
	map_link_impl::sp_map_link_impl mama, link src_link, event ev, F res_processor
) {
	std::optional<lid_type> dest_lid;
	if(auto pdest = mama->io_map_.find(src_link.id()); pdest != mama->io_map_.end())
		dest_lid = pdest->second;

	// make base actor behavior
	return [=, mama = std::move(mama), mf = mama->mf_, ev = std::move(ev), rp = std::move(res_processor)]
	(caf::event_based_actor* self) mutable {
		// register as kernel citizen if required
		if(enumval(mama->opts_ & TreeOpts::TrackWorkers))
			self->attach_functor([=] { KRADIO.release_citizen(self); });

		self->become(
			// invoke mapper & return resulting link (may be lazily evaluated)
			[=, mf = std::move(mf), ev = std::move(ev)](a_apply, const link& dest_link) mutable {
				adbg(self) << "lmapper: map " << to_string(src_link.id()) <<
					" -> " << to_string(dest_link.id()) << std::endl;

				auto res = std::optional<caf::result<link>>{};
				if(src_link) {
					if(error::eval_safe([&] { res = mf(src_link, dest_link, std::move(ev), self); }))
						res = link{};
				}
				else
					res = link{};
				return *res;
			},

			// finish exec chain: need separate handler to ensure lazy `res_link` calc finished
			[=, rp = std::move(rp)](const link& dest_link) {
				adbg(self) << "lmapper: request mapping for dest link " <<
					to_string(dest_link.id()) << " " << dest_link.name() << std::endl;

				self->request(caf::actor_cast<caf::actor>(self), caf::infinite, a_apply{}, dest_link)
				.then([=, rp = std::move(rp)](const link& res_link) mutable {
					adbg(self) << "lmapper: deliver mapped link " <<
						to_string(res_link.id()) << " " << res_link.name() << std::endl;

					// invoke result processing
					rp(lmapper_res_t{res_link, dest_link}, self);
					// [NOTE] it's essential to quit explicitly, otherwise waiting for worker will hang
					self->quit();
				});
			},

			// entry point to start calc
			[=, mama = std::move(mama)](a_ack) -> caf::result<caf::message> {
				if(dest_lid) {
					auto rp = self->make_response_promise();
					self->request(mama->out_.actor(), kernel::radio::timeout(), a_node_find{}, *dest_lid)
					.then([=](const link& dest_link) mutable {
						rp.delegate(caf::actor_cast<caf::actor>(self), dest_link);
					});
					return rp;
				}
				else
					return self->delegate(caf::actor_cast<caf::actor>(self), link{});
			},

			// support delayed eval from status waiters queue
			[=](a_apply, const node_or_errbox&) { self->send(self, a_ack{}); }
		);
	};
}

template<typename... Args>
auto spawn_lmapper_actor(map_link_actor* papa, Args&&... args) {
	auto mama = papa->spimpl<map_link_impl>();
	auto res = caf::actor{};

	if(enumval(mama->opts_ & TreeOpts::DetachedWorkers))
		res = papa->spawn<caf::detached>(
			make_lmapper_actor(std::move(mama), std::forward<Args>(args)...)
		);
	else
		res = papa->spawn(
			make_lmapper_actor(std::move(mama), std::forward<Args>(args)...)
		);

	// early register as kernel citizen if required
	if(enumval(mama->opts_ & TreeOpts::TrackWorkers))
		KRADIO.register_citizen(res.address());
	return res;
}

NAMESPACE_END()

///////////////////////////////////////////////////////////////////////////////
//  update
//
auto map_link_impl::update(map_link_actor* papa, link src_link, event ev) -> void {
	adbg(papa) << "lmapper::update " << to_string(src_link.id()) << " " << to_string(id_) << std::endl;
	// sanity - don't map self
	if(src_link.id() == id_) return;

	// define 2nd stage - process result
	auto s2_process_res = [=, src_lid = src_link.id()](const lmapper_res_t& res) mutable {
		// if nothing changed - exit
		// [NOTE] also exit if res link == src link
		// because inserting source link into output node will cause removal from source =>
		// erased event that will lead to erase from output (only one possible outcome)
		// so, prohibit this
		auto& [res_link, dest_link] = res;
		adbg(papa) << "lmapper::update::s2 " << to_string(res_link.id()) << " -> " <<
			to_string(dest_link.id()) << std::endl;
		if(res_link == dest_link || res_link.id() == src_lid) return;

		// if dest wasn't nil - erase current mapping
		auto dest_lid = dest_link.id();
		if(dest_link) {
			io_map_.erase(src_lid);
			papa->send(out_.actor(), a_node_erase(), dest_lid);
		}
		// remember result (new dest)
		if(res_link) {
			adbg(papa) << "lmapper:: inserting res link " << res_link.name() << std::endl;
			// insert or replace dest with res link
			// have to wait until insertion completes
			papa->request(
				out_.actor(), caf::infinite, a_node_insert(), res_link, InsertPolicy::AllowDupNames
			).await([=, res_id = res_link.id()](node::insert_status s) {
				adbg(papa) << "lmapper:: inserted res link " << s.second << std::endl;
				// update mapping
				if(s.first) io_map_[src_lid] = res_id;
			});
		}
	};

	if(auto lmapper = spawn_lmapper_actor(
		papa, src_link, std::move(ev),
		[papa_actor = papa->actor(), s2 = std::move(s2_process_res)]
		(lmapper_res_t res, caf::event_based_actor* worker) mutable {
			auto tr = link_transaction{[s2 = std::move(s2), res = std::move(res)]() mutable {
				s2(res);
				return perfect;
			}};
			worker->send(papa_actor, a_apply{}, std::move(tr));
		}
	))
		// if in Busy state, add to message queue, otherwise start immediately
		papa->impl.rs_apply(Req::DataNode, [&](link_impl::status_handle& S) {
			if(S.value == ReqStatus::Busy)
				S.waiters.push_back(std::move(lmapper));
			else
				papa->send(lmapper, a_ack{});
		});
}

///////////////////////////////////////////////////////////////////////////////
//  refresh
//
auto map_link_impl::refresh(map_link_actor* papa, caf::event_based_actor* rworker)
-> caf::result<node_or_errbox> {
	using namespace allow_enumops;
	adbg(papa) << "impl::refresh" << std::endl;

	auto out_leafs = links_v{};
	io_map_t io_map;
	auto mappers = std::vector<caf::actor>{};
	auto mapper_solo = engine_impl_mutex{};

	// start mappers in parallel over given leafs
	const auto map_leafs_array = [&](const links_v& in_leafs) {
		// start mappers in parallel
		std::for_each(in_leafs.begin(), in_leafs.end(), [&](const auto& src_link) {
			// sanity - don't process self
			if(src_link.id() == id_) return;

			// define mapped link processing
			auto s2_process_res = [&, src_link](const lmapper_res_t& map_res, caf::event_based_actor*) {
				const auto& [res_link, _] = map_res;
				if(res_link && res_link != src_link) {
					auto guard = std::lock_guard{mapper_solo};
					out_leafs.push_back(res_link);
					io_map[src_link.id()] = res_link.id();
				}
			};

			// spawn mapper actor & start job
			if(auto lmapper = spawn_lmapper_actor(
				papa, src_link, event{caf::actor_cast<caf::actor>(papa), {}, Event::None},
				std::move(s2_process_res)
			)) {
				rworker->send(lmapper, a_ack{});
				mappers.push_back(std::move(lmapper));
			}
		});
	};

	// invoke mapper over input leafs and save output into temp vector
	if(auto er = error::eval_safe([&] {
		if(enumval(opts_ & TreeOpts::Deep)) {
			adbg(papa) << "impl::refresh: making deep traversal" << std::endl;
			// deep walk the tree & pass each link to mapper
			walk(in_, [&](const node&, std::list<node>& subnodes, links_v& leafs) {
				// extend leafs with subnodes handles
				leafs.reserve(leafs.size() + subnodes.size());
				std::for_each(subnodes.begin(), subnodes.end(), [&](const auto& subn) {
					if(auto h = owner_handle(subn)) leafs.push_back(h);
				});
				// map resulting links
				map_leafs_array(leafs);
			}, opts_);
		}
		else {
			adbg(papa) << "impl::refresh: making plain traversal" << std::endl;
			// obtain input node's leafs
			auto in_leafs = in_.leafs();
			out_leafs.reserve(in_leafs.size());
			mappers.reserve(in_leafs.size());
			// and map 'em
			map_leafs_array(in_leafs);
		}

		// wait until all mapper workers finished
		auto waiter = caf::scoped_actor{rworker->system()};
		waiter->wait_for(std::move(mappers));
	})) return node_or_errbox{tl::unexpect, er};

	adbg(papa) << "impl::refresh: 2nd stage - inserting mapped links" << std::endl;
	using R = node_or_errbox;
	auto res = rworker->make_response_promise<R>();
	// deliver transaction error (for ex. exception inside transaction)
	auto deliver_err_res = [=](tr_result::box trb) mutable {
		// [WARN] formally, we must check OK by converting to `error`
		// OK result is delivered from node transaction lower
		if(auto tres = tr_result{std::move(trb)}; tres.err()) {
			res.deliver(R{ tl::unexpect, pack(tres.err()) });
			rworker->quit();
		}
	};

	// insert results into ouput node inside map_link transaction
	// that, in turn, runs internal transaction in output node
	rworker->request(
		papa->actor(), caf::infinite, a_apply(),
		link_transaction{[=, io_map = std::move(io_map), out_leafs = std::move(out_leafs)]() mutable {
			// update mappings
			io_map_ = std::move(io_map);
			// update output node
			rworker->request(
				node_impl::actor(out_), caf::infinite, a_apply(),
				node_transaction{[=, out_leafs = std::move(out_leafs)](bare_node dest_node) mutable {
					dest_node.clear();
					auto cnt = dest_node.insert(unsafe, std::move(out_leafs));
					res.deliver(R{ out_ });
					adbg(papa) << "refresh: inserted links count = " << cnt << std::endl;
					return perfect;
				}}
			)
			// 2. Deliver result
			.then(deliver_err_res);
			return perfect;
		}}
	).then(deliver_err_res);

	return res;
}

auto map_link_impl::refresh(map_link_actor* papa) -> caf::result<node_or_errbox> {
	// run output node refresh in separate actor
	return request_data_impl<node>(
		*papa, Req::DataNode, ReqOpts::Detached,
		[=, pimpl = std::static_pointer_cast<map_link_impl>(papa->pimpl_)]
		(caf::event_based_actor* rworker) {
			return pimpl->refresh(papa, rworker);
		}
	);
}

// default ctor installs noop mapping fn
map_link_impl::map_link_impl() :
	map_impl_base(true), mf_(noop_r<link>())
{}

auto map_link_impl::clone(link_actor*, bool deep) const -> caf::result<sp_limpl> {
	// [NOTE] output node is always brand new, otherwise a lot of questions & issues rises
	return std::make_shared<map_link_impl>(mf_, tag_, name_, in_, node::nil(), update_on_, opts_, flags_);
}

auto map_link_impl::erase(map_link_actor* self, lid_type src_lid, event) -> void {
	if(auto pdest = io_map_.find(src_lid); pdest != io_map_.end()) {
		self->send(out_.actor(), a_node_erase(), pdest->second);
		io_map_.erase(pdest);
	}
}

// [NOTE] both link -> link & node -> node impls have same type ID,
// because it must match with `map_link::type_id_()`
ENGINE_TYPE_DEF(map_link_impl, "map_link")

NAMESPACE_END(blue_sky::tree)
