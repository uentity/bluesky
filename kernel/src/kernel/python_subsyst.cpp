/// @file
/// @author uentity
/// @date 23.04.2019
/// @brief BS kernel Python subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/common.h>
#include <bs/log.h>
#include <bs/kernel/errors.h>
#include <bs/tree/node.h>

#include "kimpl.h"
#include "python_subsyst_impl.h"
#include "plugins_subsyst.h"
#include "radio_subsyst.h"

#include <pybind11/functional.h>

#include <iostream>
#include <functional>

#define WITH_KMOD \
[[maybe_unused]] auto res = pykmod(this).map([&](auto kmod_ptr) { \
[[maybe_unused]] auto& kmod = *kmod_ptr;

#define END_WITH });

#define RETURN_WITH END_WITH return res.value();

#define RETURN_WITH_FAIL(err_val) END_WITH return res.value_or(err_val);

NAMESPACE_BEGIN(blue_sky::kernel::detail)
using namespace blue_sky::detail;
namespace py = pybind11;

NAMESPACE_BEGIN()

auto pykmod(const python_subsyst* v) -> result_or_err<pybind11::module*> {
	if(v) if(auto kmod = v->py_kmod())
		return { static_cast<pybind11::module*>(kmod) };
	return tl::unexpected{Error::BadPymod};
}

std::string extract_root_name(const std::string& full_name) {
	using namespace std;

	//extract user-friendly lib name from full name
	string pyl_name = full_name;
	//cut path
	string::size_type pos = pyl_name.rfind('/');
	if(pos != string::npos) pyl_name = pyl_name.substr(pos + 1, string::npos);
	pos = pyl_name.rfind('\\');
	if(pos != string::npos) pyl_name = pyl_name.substr(pos + 1, string::npos);
	//cut extension
	pyl_name = pyl_name.substr(0, pyl_name.find('.'));
	//cut "lib" prefix if exists
	if(pyl_name.compare(0, 3, string("lib")) == 0)
		pyl_name = pyl_name.substr(3, string::npos);

	return pyl_name;
}

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  python_subsyst_impl
 *-----------------------------------------------------------------------------*/
python_subsyst_impl::python_subsyst_impl(void* kmod_ptr) :
	kmod_(nullptr)
{
	setup_py_kmod(kmod_ptr);
}

python_subsyst_impl::~python_subsyst_impl() {}

auto python_subsyst_impl::self() -> python_subsyst_impl& {
	// [NOTE] unconditional cast, because if self() is called, then instance MUST be self for sure
	return static_cast<python_subsyst_impl&>(singleton<python_subsyst>::Instance());
}

auto python_subsyst_impl::py_kmod() const -> void* { return kmod_; }

auto python_subsyst_impl::setup_py_kmod(void* kmod_ptr) -> void {
	if(!kmod_ptr) return;

	kmod_ = kmod_ptr;
	auto& kmod = *static_cast<pybind11::module*>(kmod_);

	auto& kpd = plugins_subsyst::kernel_pd();
	kmod.doc() = kpd.description;
	kpd.py_namespace = PyModule_GetName(kmod.ptr());

	// [IMPORTANT] we must clean up adapters & cache right BEFORE interpreter termination
	auto at_pyexit = py::module::import("atexit");
	at_pyexit.attr("register")(py::cpp_function{[this] {
		std::cout << "~~~ Python subsystem shutting down..." << std::endl;
		drop_adapted_cache();
		adapters_.clear();
		def_adapter_ = nullptr;

		// kick all event-based actors (normally wait until all actor handles wired into Python are released)
		{
			auto _ = py::gil_scoped_release{};
			const auto kradio = KIMPL.get_radio();
			kradio->kick_citizens();
			// stop queue while Python is still alive to gracefully finish all pending transactions
			kradio->stop_queue(true);
		}

		// force garbage collection
		auto gc = py::module::import("gc");
		gc.attr("collect")();
		gc.attr("collect")();

		std::cout << "~~~ Python subsystem down" << std::endl;
	}});
}

auto python_subsyst_impl::py_init_plugin(
	const blue_sky::detail::lib_descriptor& lib, plugin_descriptor& p_descr
) -> result_or_err<std::string> {
	return pykmod(this).and_then([&](auto kmod_ptr) -> result_or_err<std::string> {
		bs_init_py_fn init_py_fn;
		lib.load_sym("bs_init_py_subsystem", init_py_fn);
		if(!init_py_fn) return tl::make_unexpected(error{p_descr.name, kernel::Error::PythonDisabled});

		if(p_descr.py_namespace.empty())
			p_descr.py_namespace = extract_root_name(lib.fname_);

		// create submodule
		auto plugin_mod = kmod_ptr->def_submodule(p_descr.py_namespace.c_str(), p_descr.description.c_str());
		// invoke init function
		init_py_fn(&plugin_mod);

		return plugins_subsyst::kernel_pd().py_namespace + '.' + p_descr.py_namespace;
	});
}

auto python_subsyst_impl::py_add_error_closure() -> void {
WITH_KMOD
	auto err_class = (pybind11::class_<std::error_code>)kmod.attr("error_code");
	pybind11::init([](int ec) {
		return std::error_code(static_cast<Error>(ec));
	}).execute(err_class);
END_WITH
}

auto python_subsyst_impl::register_adapter(std::string obj_type_id, adapter_fn f) -> void {
	if(f)
		adapters_.insert_or_assign(std::move(obj_type_id), std::move(f));
	else {
		auto pa = adapters_.find(obj_type_id);
		if(pa != adapters_.end())
			adapters_.erase(pa);
	}
}

auto python_subsyst_impl::register_default_adapter(adapter_fn f) -> void {
	def_adapter_ = std::move(f);
}

auto python_subsyst_impl::clear_adapters() -> void {
	drop_adapted_cache();
	adapters_.clear();
	def_adapter_ = nullptr;
}

auto python_subsyst_impl::adapted_types() const -> std::vector<std::string> {
	std::vector<std::string> res(adapters_.size());
	std::transform(
		adapters_.begin(), adapters_.end(), res.begin(),
		[](const auto& A){ return A.first; }
	);
	if(def_adapter_) res.push_back("*");
	return res;
}

auto python_subsyst_impl::adapt(sp_obj source, const tree::link& L) -> py::object {
	if(!source) return py::none();

	// register link and return whether link is met for the first time
	const auto remember_link = [&](auto* data_ptr) {
		if(!lnk2obj_.try_emplace(L.id(), data_ptr).second) return false;
		// erase adapter on link delete
		L.subscribe([](const auto& ev) {
			const auto& [_, params, __] = ev;
			const auto* lid = prop::get_if<uuid>(&params, "link_id");
			if(!lid) return;

			KRADIO.enqueue(launch_async, [lid = *lid] {
				auto py_guard = py::gil_scoped_acquire();
				auto& self = python_subsyst_impl::self();
				auto cached_L = self.lnk2obj_.find(lid);
				if(cached_L == self.lnk2obj_.end()) return perfect;

				// kill cache entry if ref counter reaches zero
				if(auto cached_A = self.acache_.find(cached_L->second); cached_A != self.acache_.end()) {
					if(--cached_A->second.second == 0)
						self.acache_.erase(cached_A);
				}
				self.lnk2obj_.erase(cached_L);
				return perfect;
			});
		}, tree::Event::LinkDeleted);
		return true;
	};

	// check if adapter already created for given object
	auto* data_ptr = source.get();
	if(auto cached_A = acache_.find(data_ptr); cached_A != acache_.end()) {
		// inc ref counter for new link
		auto& [pyobj, use_count] = cached_A->second;
		if(remember_link(data_ptr))
			++use_count;
		return pyobj;
	}

	// adapt or passthrough
	const auto adapt_and_cache = [&](auto&& afn) {
		// cache adapter with ref counter = 1
		return acache_.try_emplace(
			data_ptr, afn(std::move(source)), size_t{ remember_link(data_ptr) }
		).first->second.first;
	};

	auto pf = adapters_.find(source->type_id());
	return pf != adapters_.end() ?
		adapt_and_cache(pf->second) :
		( def_adapter_ ? adapt_and_cache(def_adapter_) : py::cast(source) );
}

auto python_subsyst_impl::get_cached_adapter(const sp_obj& obj) const -> pybind11::object {
	if(auto cached_A = acache_.find(obj.get()); cached_A != acache_.end())
		return cached_A->second.first;
	return py::none();
}

auto python_subsyst_impl::drop_adapted_cache(const sp_obj& obj) -> std::size_t {
	std::size_t res = 0;
	if(!obj) {
		res = acache_.size();
		acache_.clear();
		lnk2obj_.clear();
	}
	else if(auto cached_A = acache_.find(obj.get()); cached_A != acache_.end()) {
		// clean link -> obj ptr resolver
		auto* obj_ptr = cached_A->first;
		for(auto p_lnk = lnk2obj_.begin(); p_lnk != lnk2obj_.end();) {
			if(p_lnk->second == obj_ptr)
				p_lnk = lnk2obj_.erase(p_lnk);
			else
				++p_lnk;
		}
		// clean cached adapter
		acache_.erase(cached_A);
		res = 1;
	}
	return res;
}

NAMESPACE_END(blue_sky::kernel::detail)

NAMESPACE_BEGIN(blue_sky::python)

auto py_bind_adapters(py::module& m) -> void {
	using namespace kernel::detail;
	static auto py_kernel = &python_subsyst_impl::self;

	// export adapters manip functions
	using adapter_fn = kernel::detail::python_subsyst_impl::adapter_fn;

	m.def("register_adapter", [](std::string obj_type_id, adapter_fn f) {
			py_kernel().register_adapter(std::move(obj_type_id), std::move(f));
		}, "obj_type_id"_a, "adapter_fn"_a, "Register adapter for specified BS type"
	);
	m.def("register_default_adapter", [](adapter_fn f) {
			py_kernel().register_default_adapter(std::move(f));
		}, "adapter_fn"_a, "Register default adapter for all BS types with no adapter registered"
	);
	m.def("adapted_types", []() { return py_kernel().adapted_types(); },
		"Return list of types with registered adapters ('*' denotes default adapter)"
	);
	m.def("adapt", [](sp_obj source, const tree::link& L) {
			return py_kernel().adapt(std::move(source), L);
		}, "source"_a, "lnk"_a,
		"Make adapter for given object"
	);

	m.def("clear_adapters", []() { py_kernel().clear_adapters(); },
		"Remove all adapters (including default) for BS types"
	);
	m.def("drop_adapted_cache", [](const sp_obj& obj) {
			return py_kernel().drop_adapted_cache(obj);
		}, "obj"_a = nullptr,
		"Clear cached adapter for given object (or drop all cached adapters if object is None)"
	);
	m.def("get_cached_adapter", [](const sp_obj& obj) {
			return py_kernel().get_cached_adapter(obj);
		},
		"obj"_a, "Get cached adapter for given object (if created before, otherwise None)"
	);

	m.def("enqueue", [](py::function f) {
		KRADIO.enqueue(launch_async, [f = std::move(f)] {
			f();
			return perfect;
		});
	}, "f"_a, "Async run Python function in BS queue");
}

NAMESPACE_END(blue_sky::python)
