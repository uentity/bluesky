/// @file
/// @author uentity
/// @date 22.11.2019
/// @brief Implements data request
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "request_impl.h"

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
using namespace allow_enumops;

auto link_actor::data_ex(obj_processor_f cb, ReqOpts opts) -> void {
	request_impl(
		*this, Req::Data, opts,
		[Limpl = pimpl_] { return Limpl->data(); },
		std::move(cb)
	);
}

auto link_actor::data_node_ex(node_processor_f cb, ReqOpts opts) -> void {
	data_node_request(*this, opts, std::move(cb));
}

auto cached_link_actor::data_ex(obj_processor_f cb, ReqOpts opts) -> void {
	request_impl(
		*this, Req::Data, opts | ReqOpts::HasDataCache,
		[Limpl = pimpl_] { return Limpl->data(); },
		std::move(cb)
	);
}

auto cached_link_actor::data_node_ex(node_processor_f cb, ReqOpts opts) -> void {
	data_node_request(*this, opts | ReqOpts::HasDataCache, std::move(cb));
}

auto fast_link_actor::data_ex(obj_processor_f cb, ReqOpts opts) -> void {
	// directly invoke `impl.data()` regardless of any options
	request_impl<false>(
		*this, Req::Data, opts | ReqOpts::DirectInvoke,
		[Limpl = pimpl_] { return Limpl->data(); },
		std::move(cb)
	);
}

auto fast_link_actor::data_node_ex(node_processor_f cb, ReqOpts opts) -> void {
	data_node_request<false>(*this, opts | ReqOpts::DirectInvoke, std::move(cb));
}

NAMESPACE_END(blue_sky::tree)
