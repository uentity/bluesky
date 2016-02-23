/// @file
/// @author Sergey Miryanov
/// @date 07.07.2009
/// @brief Scribers for log channels
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_LOG_SCRIBERS_H_
#define BS_LOG_SCRIBERS_H_

#include "bs_log_stream.h"
#include <fstream>

namespace blue_sky {
namespace log {
namespace detail {

	class BS_API cout_scriber : public bs_stream {
	public:

    cout_scriber (const std::string &name)
    : bs_stream (name)
    {
    }

		void write(const std::string &str) const;
	};

	class BS_API file_scriber : public bs_stream {
	public:
		typedef smart_ptr< std::fstream > sp_fstream;

		file_scriber(const std::string &name, const std::string &filename, std::ios_base::openmode mode);
		void write(const std::string &str) const;

	private:
		sp_fstream file;
	};

} // namespace detail
} // namespace log
} // namespace blue_sky


#endif  // #ifndef BS_LOG_SCRIBERS_H_

