/// @file
/// @author uentity
/// @date 22.06.2018
/// @brief Implementation of BS tree-related serialization
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/atoms.h>
#include <bs/kernel/radio.h>
#include <bs/serialize/serialize.h>
#include <bs/serialize/tree.h>
#include "../tree/actor_common.h"

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::on_serialized_f)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::sp_link)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::error::box)

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
/*-----------------------------------------------------------------------------
 *  tree serialization actor impl
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()
///////////////////////////////////////////////////////////////////////////////
//  Tree FS archive
//
auto save_fs(const sp_link& root, const std::string& filename) -> error::box {
	// collect all errors happened
	auto errs = std::vector<error>{};
	if(auto er = error::eval_safe([&] {
		auto ar = tree_fs_output(filename);
		ar(root);
		errs = ar.wait_objects_saved(infinite);
	}))
		errs.push_back(er);

	std::string reduced_er;
	for(const auto& er : errs) {
		if(er) {
			if(!reduced_er.empty()) reduced_er += '\n';
			reduced_er += er.what();
		}
	}
	return reduced_er.empty() ? success() : error::quiet(reduced_er);
}

auto load_fs(sp_link& root, const std::string& filename) -> error::box {
	return error::eval_safe([&] {
		auto ar = tree_fs_input(filename);
		ar(root);
		ar.serializeDeferments();
	});
}

///////////////////////////////////////////////////////////////////////////////
//  generic archives
//
auto save_generic(const sp_link& root, const std::string& filename, TreeArchive ar) -> error::box {
return error::eval_safe([&]() -> error {
	// open file for writing
	std::ofstream fs(
		filename,
		std::ios::out | std::ios::trunc | (ar == TreeArchive::Binary ? std::ios::binary : std::ios::openmode())
	);
	fs.exceptions(fs.failbit | fs.badbit);

	// dump link to JSON archive
	if(ar == TreeArchive::Binary) {
		cereal::PortableBinaryOutputArchive ja(fs);
		ja(root);
	}
	else {
		cereal::JSONOutputArchive ja(fs);
		ja(root);
	}
	return perfect;
}); }

auto load_generic(sp_link& root, const std::string& filename, TreeArchive ar) -> error::box {
return error::eval_safe([&]() -> error {
	// open file for reading
	std::ifstream fs(
		filename,
		std::ios::in | (ar == TreeArchive::Binary ? std::ios::binary : std::ios::openmode())
	);
	fs.exceptions(fs.failbit | fs.badbit);

	// load link from JSON archive
	if(ar == TreeArchive::Binary) {
		cereal::PortableBinaryInputArchive ja(fs);
		ja(root);
	}
	else {
		cereal::JSONInputArchive ja(fs);
		ja(root);
	}
	return perfect;
}); }

///////////////////////////////////////////////////////////////////////////////
//  serial ator impl
//
struct serial_state {
	error::box er;
	caf::response_promise erp;
	bool finished = false;
};

enum class Serial { Save, Load };

template<Serial Mode>
static auto serial_actor(
	caf::stateful_actor<serial_state>* self, sp_link root, std::string filename, TreeArchive ar,
	on_serialized_f cb
) -> caf::behavior {
	// start main work
	self->send(self, a_hi());

	return {
		[self, ar, r = std::move(root), filename = std::move(filename), cb = std::move(cb)](a_hi) mutable {
			auto& S = self->state;
			// launch work
			if constexpr(Mode == Serial::Save)
				S.er = ar == TreeArchive::FS ? save_fs(r, filename) : save_generic(r, filename, ar);
			else
				S.er = ar == TreeArchive::FS ? load_fs(r, filename) : load_generic(r, filename, ar);
			self->state.finished = true;
			// invoke callback
			// [NOTE] it's essential to invoke callback BEFORE error is delivered
			if(cb) cb(std::move(r), error::unpack(S.er));
			// deliver error
			S.erp.deliver(std::move(S.er));
		},
		
		[self](a_ack) -> caf::result<error::box> {
			auto& S = self->state;
			if(S.finished)
				return S.er;
			else {
				S.erp = self->make_response_promise<error::box>();
				return S.erp;
			}
		}
	};
}

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  tree save impl
 *-----------------------------------------------------------------------------*/
auto save_tree(sp_link root, std::string filename, TreeArchive ar, timespan wait_for) -> error {
	// launch worker in detached actor
	auto Af = caf::make_function_view(
		system().spawn<caf::spawn_options::detach_flag> (
			serial_actor<Serial::Save>, std::move(root), std::move(filename), ar, on_serialized_f{}
		),
		wait_for == infinite ? caf::infinite : caf::duration{wait_for}
	);
	// wait for result
	if(auto res = actorf<error::box>(Af, a_ack()); res)
		return error::unpack(res.value());
	else
		return std::move(res.error());
}

BS_API auto save_tree(
	on_serialized_f cb, sp_link root, std::string filename, TreeArchive ar
) -> void {
	// [NOTE] spawn save in detached actor to prevent starvation
	system().spawn<caf::spawn_options::detach_flag> (
		serial_actor<Serial::Save>, std::move(root), std::move(filename), ar, std::move(cb)
	);
}

/*-----------------------------------------------------------------------------
 *  tree load impl
 *-----------------------------------------------------------------------------*/
auto load_tree(std::string filename, TreeArchive ar) -> result_or_err<sp_link> {
	// launch worker in detached actor
	sp_link root;
	auto Af = caf::make_function_view(
		system().spawn<caf::spawn_options::detach_flag> (
			serial_actor<Serial::Load>, root, std::move(filename), ar,
			// pass callback that passes deserialized root from actor into this context
			[&](sp_link res, error) { root = std::move(res); }
		),
		caf::infinite
	);
	// wait for result
	error::box res_erb;
	if(auto res = actorf<error::box>(Af, a_ack()); res)
		res_erb = std::move(res.value());
	else
		res_erb = res.error();

	if(auto res_er = error::unpack(std::move(res_erb)))
		return tl::make_unexpected(std::move(res_er));
	else
		return root;
}

auto load_tree(
	on_serialized_f cb, std::string filename, TreeArchive ar
) -> void {
	// [NOTE] spawn save in detached actor to prevent starvation
	system().spawn<caf::spawn_options::detach_flag> (
		serial_actor<Serial::Load>, sp_link{}, std::move(filename), ar, std::move(cb)
	);
}

NAMESPACE_END(blue_sky::tree)
