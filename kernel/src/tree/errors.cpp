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

			case Error::WrongOrderSize :
				return "Size of given keys set don't match size of node";

			case Error::EmptyPath :
				return "Path is empty";

			case Error::PathNotExists :
				return "Path doesn't exist";

			case Error::PathNotDirectory :
				return "Path is not a directory";

			case Error::CantReadFile :
				return "Can't open file for reading";

			case Error::CantWriteFile :
				return "Can't open file for writing";

			case Error::LinkWasntStarted :
				return "BS tree::link save/load wasn't started";

			case Error::NodeWasntStarted :
				return "BS tree::node save/load wasn't started";

			case Error::MissingFormatter :
				return "Formatter isn't installed for given object type";

			case Error::CantMakeFilename :
				return "Couldn't generate unique filename";

			case Error::WrongLinkCast :
				return "Wrong link cast";

			default:
				return "";
			}
		}
	};

	return { static_cast<int>(e), tree_domain::self() };
}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

