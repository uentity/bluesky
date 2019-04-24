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

#define WITH_KMOD \
[[maybe_unused]] auto res = pykmod(this).map([&](auto kmod_ptr) { \
[[maybe_unused]] auto& kmod = *kmod_ptr;

#define END_WITH });

#define RETURN_WITH END_WITH return res.value();

#define RETURN_WITH_FAIL(err_val) END_WITH return res.value_or(err_val);

NAMESPACE_BEGIN(blue_sky::kernel::detail)
using namespace blue_sky::detail;

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

NAMESPACE_END(blue_sky::kernel::detail)
