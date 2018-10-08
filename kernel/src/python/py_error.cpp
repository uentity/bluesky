/// @file
/// @author uentity
/// @date 28.02.2018
/// @brief Python bindings for BS error subsystem
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

void py_bind_error(py::module& m) {
	/*-----------------------------------------------------------------------------
	 *  Error codes
	 *-----------------------------------------------------------------------------*/
	py::enum_<Error>(m, "Error")
		.value("OK", Error::OK)
		.value("Happened", Error::Happened)
	;

	py::enum_<KernelError>(m, "KernelError")
		.value("OK", KernelError::OK)
		.value("CantLoadDLL", KernelError::CantLoadDLL)
		.value("CantUnloadDLL", KernelError::CantUnloadDLL)
		.value("CantRegisterType", KernelError::CantRegisterType)
		.value("TypeIsNil", KernelError::TypeIsNil)
		.value("TypeAlreadyRegistered", KernelError::TypeAlreadyRegistered)
		.value("CantCreateLogger", KernelError::CantCreateLogger)
	;

	py::enum_<tree::Error>(m, "TreeError")
		.value("OK", tree::Error::OK)
		.value("OKOK", tree::Error::OKOK)
		.value("EmptyData", tree::Error::EmptyData)
		.value("NotANode", tree::Error::NotANode)
		.value("LinkExpired", tree::Error::LinkExpired)
		.value("UnboundSymLink", tree::Error::UnboundSymLink)
		.value("LinkBusy", tree::Error::LinkBusy)
		.value("NoFusionBridge", tree::Error::NoFusionBridge)
		.value("KeyMismatch", tree::Error::KeyMismatch)
	;

	/*-----------------------------------------------------------------------------
	 *  Error classes
	 *-----------------------------------------------------------------------------*/
	// bind std::error_code
	py::class_<std::error_code>(m, "error_code")
		// empty ctor results in Error::Happened
		.def(py::init([]() { return std::error_code(Error::Happened); }))
		// construct from known error enums -- plugins can add ctors like this
		.def(py::init<Error>())
		.def(py::init<KernelError>())
		.def(py::init<tree::Error>())

		.def_property_readonly("value", &std::error_code::value, "Get numeric error code")
		.def_property_readonly(
			"category",
			[](const std::error_code& c) { return c.category().name(); },
			"Get error code category"
		)
		.def_property_readonly(
			"message", &std::error_code::message, "Explanatory string about this error code"
		)
		.def(py::self == py::self)
		.def(py::self != py::self)
		.def(py::self < py::self)
		.def("__bool__", [](const std::error_code& c){ return (bool)c; })
		.def("__hash__", [](const std::error_code& c){ return std::hash<std::error_code>{}(c); })
		.def("__repr__", [](const std::error_code& c){
			return fmt::format("[{}] [{}] {}", c.category().name(), c.value(), c.message());
		})
	;
	// implcitly construct error_code from corresponding enum value
	py::implicitly_convertible<Error, std::error_code>();
	py::implicitly_convertible<KernelError, std::error_code>();
	py::implicitly_convertible<tree::Error, std::error_code>();

	// bind blue_sky::error
	py::class_<error>(m, "error")
		// ctors
		// from optional error_code
		.def(py::init<const std::error_code>(), "code"_a = Error::Happened)
		// message + optional error_code
		.def(py::init<const std::string, const std::error_code>(),
			"message"_a, "code"_a = Error::Happened
		)
		// int code -- disable this ctor, because it will be used if some error enum value passed
		//.def(py::init<int>())

		// quiet
		.def_static("quiet",
			&error::quiet<const std::string, const std::error_code&>,
			"message"_a, "code"_a = Error::OK
		)
		.def_static("quiet",
			&error::quiet<const std::error_code&>,
			"code"_a = Error::OK
		)
		// [TODO]: implement better solution
		// allow quiet construct from int, but this results in that all error enum values get
		// auto-converted to int and passed as int here
		.def_static("quiet", &error::quiet<int>, "code"_a)

		// other methods
		.def_property_readonly("domain", &error::domain, "Get error domain (error_code::category)")
		.def("dump", &error::dump, "Log current error")
		.def("__repr__", &error::to_string)
		// export code memeber
		.def_readonly("code", &error::code, "Access code of this error")
		// error message
		.def_property_readonly("what", &error::what, "Get error message")
		.def_property_readonly("ok", &error::ok, "Test if no error happened (successfull op)")
	;
	py::implicitly_convertible<Error, error>();

	m.def("test_code", [](int c) { bsout() << c << log::end; });
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

