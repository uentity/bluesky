/// @date 05.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "map_engine.h"

#include <bs/meta.h>
#include <bs/tree/tree.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>
#include <bs/log.h>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  map_link_impl
 *-----------------------------------------------------------------------------*/
map_link_impl::map_link_impl(
	std::string name, link_mapper_f mf, link_or_node input, link_or_node output,
	Event update_on, TreeOpts opts, Flags f
) : super(std::move(name), f), mf_(std::move(mf)), update_on_(update_on), opts_(opts)
{
	static const auto extract_node = [](node& lhs, auto&& rhs) {
		visit(meta::overloaded{
			[&](link L) { lhs = L ? L.data_node() : node::nil(); },
			[&](node N) { lhs = N; }
		}, std::move(rhs));
	};

	extract_node(in_, std::move(input));
	extract_node(out_, std::move(output));
	// ensure we have valid output node
	if(!out_ || in_ == out_) out_ = node();

	// set initial status values
	rs_reset(Req::Data, ReqReset::Always, ReqStatus::Error);
	rs_reset(Req::DataNode, ReqReset::Always, ReqStatus::Void);
}

auto map_link_impl::spawn_actor(sp_limpl Limpl) const -> caf::actor {
	return spawn_lactor<map_link_actor>(std::move(Limpl));
}

auto map_link_impl::clone(bool deep) const -> sp_limpl {
	// [NOTE] output node is always brand new, otherwise a lot of questions & issues rises
	return std::make_shared<map_link_impl>(name_, mf_, in_, node::nil(), update_on_, opts_, flags_);
}

auto map_link_impl::propagate_handle() -> node_or_err {
	// if output node doesn't have handle (for ex new node created in ctor) - set it to self
	if(!out_.handle())
		super::propagate_handle(out_);
	return out_;
}

// Data request always return error
auto map_link_impl::data() -> obj_or_err { return unexpected_err_quiet(Error::EmptyData); }
auto map_link_impl::data(unsafe_t) const -> sp_obj { return nullptr; }

auto map_link_impl::data_node(unsafe_t) const -> node { return out_; }

auto map_link_impl::erase(map_link_actor* self, lid_type src_lid) -> void {
	if(auto pdest = io_map_.find(src_lid); pdest != io_map_.end()) {
		self->send(out_.actor(), a_node_erase(), pdest->second);
		io_map_.erase(pdest);
	}
}

///////////////////////////////////////////////////////////////////////////////
//  update
//
template<bool SearchDest, typename... Args>
auto do_refresh_s1(map_link_actor* self, TreeOpts opts, Args&&... args) {
	using namespace allow_enumops;

	if constexpr(SearchDest) {
		if(enumval(opts & TreeOpts::DetachedWorkers))
			return anon_request_result<caf::detached>(std::forward<Args>(args)...);
		else
			return anon_request_result(std::forward<Args>(args)...);
	}
	else {
		if(enumval(opts & TreeOpts::DetachedWorkers))
			return self->spawn<caf::detached>(std::forward<Args>(args)...);
		else
			return self->spawn(std::forward<Args>(args)...);
	}	
}

auto map_link_impl::update(map_link_actor* self, link src_link) -> void {
	adbg(self) << "impl::update " << to_string(src_link.id()) << " " << to_string(id_) << std::endl;
	// sanity - don't process self
	if(src_link.id() == id_) return;

	// define 1st stage - invoke mapper function with source & dest links
	auto s1_invoke_mapper = [self, this, src_link](link dest_link) {
		adbg(self) << "impl::update::s1 invoke mapper on dest " << to_string(dest_link.id()) << std::endl;
		auto res = link{};
		if(src_link)
			error::eval_safe([&] { res = mf_(src_link, dest_link); });
		return std::pair{res, dest_link};
	};

	// run 1st stage in separate actor
	auto s1_actor = [&]() -> caf::actor {
		if(auto pdest = io_map_.find(src_link.id()); pdest != io_map_.end())
			return do_refresh_s1<true>(
				self, opts_, out_.actor(), caf::infinite, false, std::move(s1_invoke_mapper),
				a_node_find(), pdest->second
			);
		else
			return do_refresh_s1<false>(self, opts_, [f = std::move(s1_invoke_mapper)] {
				return caf::behavior{
					[f = std::move(f)](a_ack) mutable { return f(link{}); }
				};
			});
	}();

	// define 2nd stage - process result
	auto s2_process_res = [=, src_lid = src_link.id()](std::pair<link, link> res) mutable {
		// if nothing changed - exit
		// [NOTE] also exit if res link == src link
		// because inserting source link into output node will cause removal from source =>
		// erased event that will lead to erase from output (only one possible outcome)
		// so, prohibit this
		auto& [res_link, dest_link] = res;
		adbg(self) << "impl::update::s2 " << to_string(res_link.id()) << " -> " <<
			to_string(dest_link.id()) << std::endl;
		if(res_link == dest_link || res_link.id() == src_lid) return;

		// if dest wasn't nil - erase current mapping
		auto dest_lid = dest_link.id();
		if(dest_link) {
			io_map_.erase(src_lid);
			self->send(out_.actor(), a_node_erase(), dest_lid);
		}
		// remember result (new dest)
		if(res_link) {
			adbg(self) << "impl:: inserting res link " << res_link.name() << std::endl;
			// insert or replace dest with res link
			// have to wait until insertion completes
			self->request(
				out_.actor(), caf::infinite, a_node_insert(), res_link, InsertPolicy::AllowDupNames
			).await([=, res_id = res_link.id()](node::insert_status s) {
				// update mapping
				if(s.first) io_map_[src_lid] = res_id;
			});
		}
	};
	// run 2nd stage after 1st completes
	self->request(s1_actor, caf::infinite, a_ack())
	.then(std::move(s2_process_res));
}

///////////////////////////////////////////////////////////////////////////////
//  refresh
//
auto map_link_impl::refresh(map_link_actor* self, caf::event_based_actor* rworker)
-> caf::result<node_or_errbox> {
	using namespace allow_enumops;
	adbg(self) << "impl::refresh" << std::endl;

	auto out_leafs = links_v{};
	io_map_t io_map;

	// body of actor that will ivoke mapper for given link on `a_ack` message
	auto mapper_solo = engine_impl_mutex{};
	const auto mapper_actor = [&](caf::event_based_actor* self, link src_link) {
		// immediately start job
		self->send(self, a_ack());
		self->become({ [&, self, src_link](a_ack) {
			auto res_link = link{};
			error::eval_safe([&] { res_link = mf_(src_link, link{}); });
			if(res_link && res_link != src_link) {
				auto guard = std::lock_guard{mapper_solo};
				out_leafs.push_back(res_link);
				io_map[src_link.id()] = res_link.id();
			}
			// it's essential to quit when done, otherwise waiting later will hang
			self->quit();
		} });
	};

	// start mappers in parallel over given leafs
	auto mappers = std::vector<caf::actor>{};
	const auto map_leafs_array = [&](const links_v& in_leafs) {
		// start mappers in parallel
		std::for_each(in_leafs.begin(), in_leafs.end(), [&](const auto& src_link) {
			using namespace allow_enumops;
			// sanity - don't process self
			if(src_link.id() == id_) return;

			if(enumval(opts_ & TreeOpts::DetachedWorkers))
				mappers.push_back(rworker->spawn<caf::detached>(mapper_actor, src_link));
			else
				mappers.push_back(rworker->spawn(mapper_actor, src_link));
		});
	};

	// invoke mapper over input leafs and save output into temp vector
	if(auto er = error::eval_safe([&] {
		if(enumval(opts_ & TreeOpts::Deep)) {
			adbg(self) << "impl::refresh: making deep traversal" << std::endl;
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
			adbg(self) << "impl::refresh: making plain traversal" << std::endl;
			// obtain input node's leafs
			auto in_leafs = in_.leafs();
			out_leafs.reserve(in_leafs.size());
			mappers.reserve(in_leafs.size());
			// and map 'em
			map_leafs_array(in_leafs);
		}

		// wait until all mapper workers finished
		auto waiter = caf::scoped_actor{kernel::radio::system()};
		waiter->wait_for(std::move(mappers));
	})) return node_or_errbox{tl::unexpect, er};

	adbg(self) << "impl::refresh: delegating to self" << std::endl;
	using R = node_or_errbox;
	auto res = rworker->make_response_promise<R>();
	// deliver transaction error (for ex. exception inside transaction)
	auto deliver_err_res = [=](const error::box& erb) mutable {
		// [WARN] formally, we must check OK by converting to `error`
		// OK result is delivered from node transaction lower
		if(erb.ec) {
			res.deliver(R{ tl::unexpect, std::move(erb) });
			rworker->quit();
		}
	};

	// insert results into ouput node inside map_link transaction
	// that, in turn, runs internal transaction in output node
	rworker->request(
		self->actor(), caf::infinite, a_apply(),
		simple_transaction{[=, io_map = std::move(io_map), out_leafs = std::move(out_leafs)]() mutable {
			// update mappings
			io_map_ = std::move(io_map);
			// update output node
			rworker->request(
				node_impl::actor(out_), caf::infinite, a_apply(),
				node_transaction{[=, out_leafs = std::move(out_leafs)](bare_node dest_node) mutable {
					dest_node.clear();
					auto cnt = dest_node.insert(unsafe, std::move(out_leafs));
					res.deliver(R{ out_ });
					rworker->quit();
					adbg(this) << "refresh: inserted links count = " << cnt << std::endl;
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

ENGINE_TYPE_DEF(map_link_impl, "map_link")

NAMESPACE_END(blue_sky::tree)
