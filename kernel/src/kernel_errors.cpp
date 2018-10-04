/// @file
/// @author uentity
/// @date 27.02.2018
/// @brief Categories for kernel error enums
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel_errors.h>
//#include <bs/detail/lib_descriptor.h>

NAMESPACE_BEGIN(blue_sky)

BS_API std::error_code make_error_code(KernelError e) {
	// error_category for kernel errors
	static const struct : std::error_category {
		const char* name() const noexcept override {
			return "blue_sky::kernel";
		}

		std::string message(int ec) const override {
			switch(static_cast<KernelError>(ec)) {
			case KernelError::CantLoadDLL:
				return std::string("Can't load DLL");

			case KernelError::CantUnloadDLL:
				return std::string("Can't unload DLL");

			case KernelError::CantRegisterType:
				return "Type cannot be registered";

			default:
				return "";
			}
		}
	} kernel_category;

	return { static_cast<int>(e), kernel_category };
}

NAMESPACE_END(blue_sky)

