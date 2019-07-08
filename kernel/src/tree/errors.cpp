/// @file
/// @author uentity
/// @date 19.09.2018
/// @brief BS tree error codes impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/errors.h>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

BS_API std::error_code make_error_code(Error e) {
	struct tree_domain : public error::category<tree_domain> {
		const char* name() const noexcept override {
			return "blue_sky::tree";
		}

		std::string message(int ev) const override {
			switch(static_cast<Error>(ev)) {

			case Error::EmptyData:
				return "Empty data";

			case Error::EmptyInode:
				return "inode is missing";

			case Error::NotANode:
				return "Not a node";

			case Error::LinkExpired:
				return "Link is expired";

			case Error::UnboundSymLink:
				return "Unbound sym link";

			case Error::LinkBusy:
				return "Link is busy";

			case Error::NoFusionBridge:
				return "Fusion bridge isn't set";

			case Error::KeyMismatch :
				return "Given key is not found";

			default:
				return "";
			}
		}
	};

	return { static_cast<int>(e), tree_domain::self() };
}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

