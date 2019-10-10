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

#include "kimpl.h"
#include "python_subsyst_impl.h"
#include "plugins_subsyst.h"

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

constexpr auto sp_obj_hash = std::hash<sp_obj>{};

NAMESPACE_END()

python_subsyst_impl::python_subsyst_impl(void* kmod_ptr) {
	setup_py_kmod(kmod_ptr);
}

auto python_subsyst_impl::py_kmod() const -> void* { return kmod_; }

auto python_subsyst_impl::setup_py_kmod(void* kmod_ptr) -> void {
	if(kmod_ptr) {
		kmod_ = kmod_ptr;
		auto& kmod = *static_cast<pybind11::module*>(kmod_);

		auto& kpd = plugins_subsyst::kernel_pd();
		kmod.doc() = kpd.description;
		kpd.py_namespace = PyModule_GetName(kmod.ptr());
	}
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
	auto solo = std::lock_guard{ acache_guard_ };
	if(f)
		adapters_.insert_or_assign(std::move(obj_type_id), std::move(f));
	else {
		auto pa = adapters_.find(obj_type_id);
		if(pa != adapters_.end())
			adapters_.erase(pa);
	}
}

auto python_subsyst_impl::register_default_adapter(adapter_fn f) -> void {
	auto solo = std::lock_guard{ acache_guard_ };
	def_adapter_ = std::move(f);
}

auto python_subsyst_impl::clear_adapters() -> void {
	auto solo = std::lock_guard{ acache_guard_ };
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

auto python_subsyst_impl::adapt(sp_obj source) -> py::object {
	if(!source) return py::none();
	auto solo = std::lock_guard{ acache_guard_ };

	// check if adpater already created for given object
	auto cached_A = acache_.find(sp_obj_hash(source));
	if(cached_A != acache_.end())
		return cached_A->second;
	// adapt or passthrough
	auto pf = adapters_.find(source->type_id());
	auto&& obj = std::move(source);
	auto A = pf != adapters_.end() ?
		pf->second(obj) :
		( def_adapter_ ? def_adapter_(obj) : py::cast(obj) );

	// cache adapter instance
	acache_[sp_obj_hash(source)] = A;
	return A;
}

auto python_subsyst_impl::get_cached_adapter(const sp_obj& obj) const -> pybind11::object {
	if(auto cached_A = acache_.find(sp_obj_hash(obj)); cached_A != acache_.end())
		return cached_A->second;
	return py::none();
}

auto python_subsyst_impl::drop_adapted_cache(const sp_obj& obj) -> std::size_t {
	auto solo = std::lock_guard{ acache_guard_ };
	std::size_t res = 0;
	if(!obj) {
		res = acache_.size();
		acache_.clear();
	}
	else if(auto cached_A = acache_.find(sp_obj_hash(obj)); cached_A != acache_.end()) {
		acache_.erase(cached_A);
		res = 1;
	}
	return res;
}

auto python_subsyst_impl::self() -> python_subsyst_impl& {
	// [NOTE] unconditional cast, because if self() is called, then instance MUST be self for sure
	return static_cast<python_subsyst_impl&>(singleton<python_subsyst>::Instance());
}

NAMESPACE_END(blue_sky::kernel::detail)
