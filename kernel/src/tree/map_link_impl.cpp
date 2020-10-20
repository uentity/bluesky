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
			[&](link L) { lhs = L.data_node(); },
			[&](node N) { lhs = N; }
		}, std::move(rhs));
	};

	extract_node(in_, std::move(input));
	extract_node(out_, std::move(output));
	// ensure we have valid output node
	if(!out_ || in_ == out_) out_ = node();
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

auto map_link_impl::update(map_link_actor* self, link src_link) -> void {
	// define 1st stage - invoke mapper function with source & dest links
	auto s1_invoke_mapper = [this, src_link](link dest_link) {
		auto res = link{};
		if(src_link)
			error::eval_safe([&] { res = mf_(src_link, dest_link); });
		return std::pair{res, dest_link};
	};

	// run 1st stage in separate actor
	// [TODO] replace detached flag with dedicated solution for Python mappers
	auto s1_actor = [&]() -> caf::actor {
		if(auto pdest = io_map_.find(src_link.id()); pdest != io_map_.end())
			return anon_request_result<caf::detached>(
				out_.actor(), caf::infinite, false, std::move(s1_invoke_mapper),
				a_node_find(), pdest->second
			);
		else
			return self->spawn<caf::detached>([f = std::move(s1_invoke_mapper)] {
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
		if(res_link == dest_link || res_link.id() == src_lid) return;

		// if dest wasn't nil - erase current mapping
		auto dest_lid = dest_link.id();
		if(dest_link) {
			io_map_.erase(src_lid);
			self->send(out_.actor(), a_node_erase(), dest_lid);
		}
		// remember result (new dest)
		if(res_link) {
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

auto map_link_impl::erase(map_link_actor* self, lid_type src_lid) -> void {
	if(auto pdest = io_map_.find(src_lid); pdest != io_map_.end()) {
		io_map_.erase(pdest);
		self->send(out_.actor(), a_node_erase(), src_lid);
	}
}

// [NOTE] assume executing inside output node transaction
auto map_link_impl::refresh(map_link_actor* self, caf::event_based_actor* rworker)
-> caf::result<error::box> {
	auto out_leafs = links_v{};
	io_map_t io_map;

	// invoke mapper over input leafs and save output into temp vector
SCOPE_EVAL_SAFE
	auto in_leafs = in_.leafs();
	out_leafs.reserve(in_leafs.size());
	auto mappers = std::vector<caf::actor>{};
	mappers.reserve(in_leafs.size());
	auto worker_solo = engine_impl_mutex{};

	// start mappers in parallel
	for(auto& src_link : in_leafs) {
		// spawn mapper actor that does work on `a_ack` message
		// [TODO] replace detached flag with dedicated solution for Python mappers
		mappers.push_back(rworker->spawn<caf::detached>(
			[&](caf::event_based_actor* self, link src_link) {
				// immediately start job
				self->send(self, a_ack());
				self->become({ [&, self, src_link](a_ack) {
					auto res_link = link{};
					error::eval_safe([&] { res_link = mf_(src_link, link{}); });
					if(res_link) {
						auto guard = std::lock_guard{worker_solo};
						out_leafs.push_back(res_link);
						io_map[src_link.id()] = res_link.id();
					}
					// it's essential to quit when done, otherwise waiting later will hang
					self->quit();
				} });
			}, src_link
		));
	}
	// wait until they finished
	auto waiter = caf::scoped_actor{kernel::radio::system()};
	waiter->wait_for(std::move(mappers));
RETURN_SCOPE_ERR

	// insert results into ouput node inside map_link transaction
	// that, in turn, runs internal transaction in output node
	return rworker->delegate(self->actor(), a_apply(), simple_transaction(
		[=, io_map = std::move(io_map), out_leafs = std::move(out_leafs)]() mutable {
			// update mappings
			io_map_ = std::move(io_map);
			// update output node
			return out_.apply([=] {
				auto dest_node = out_.bare();
				dest_node.clear();
				for(auto& res_link : out_leafs)
					dest_node.insert(res_link, InsertPolicy::AllowDupNames);
				return perfect;
			});
		}
	));
}

ENGINE_TYPE_DEF(map_link_impl, "map_link")

NAMESPACE_END(blue_sky::tree)
