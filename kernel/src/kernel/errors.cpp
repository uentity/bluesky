/// @file
/// @author uentity
/// @date 27.02.2018
/// @brief Categories for kernel error enums
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/errors.h>

NAMESPACE_BEGIN(blue_sky::kernel)

BS_API std::error_code make_error_code(Error e) {
	// error_category for kernel errors
	static const struct : std::error_category {
		const char* name() const noexcept override {
			return "blue_sky::kernel";
		}

		std::string message(int ec) const override {
			switch(static_cast<Error>(ec)) {
			case Error::CantLoadDLL:
				return std::string("Can't load DLL");

			case Error::CantUnloadDLL:
				return std::string("Can't unload DLL");

			case Error::CantRegisterType:
				return "Type cannot be registered";

			case Error::CantCreateLogger:
				return "Cannot create logger";

			case Error::BadBSplugin:
				return "Not a BlueSky plugin";

			case Error::BadPluginDescriptor:
				return "Incorrect plugin descriptor";

			case Error::PluginAlreadyRegistered:
				return "Plugin is already registered";

			case Error::PluginRegisterFail:
				return "Error during plugin registering";

			case Error::PythonDisabled:
				return "No Python support found in this module";

			case Error::BadPymod:
				return "BS Python module isn't initialized";

			default:
				return "";
			}
		}
	} kernel_category;

	return { static_cast<int>(e), kernel_category };
}

NAMESPACE_END(blue_sky::kernel)

