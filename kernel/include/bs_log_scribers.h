/**
 * \file bs_log_scribers.h
 * \brief scribers for log channels
 * \author Sergey Miryanov
 * \date 07.07.2009
 * */
#ifndef BS_LOG_SCRIBERS_H_
#define BS_LOG_SCRIBERS_H_

#include "bs_log_stream.h"

namespace blue_sky {
namespace log {
namespace detail {

	class BS_API cout_scriber : public bs_stream {
	public:
		void write(const std::string &str) const;
	};

	class BS_API file_scriber : public bs_stream {
	public:
		typedef smart_ptr< std::fstream > sp_fstream;

		file_scriber() {}
	  file_scriber(const file_scriber &src) : bs_stream() { *this = src; }

		file_scriber(const std::string &filename, std::ios_base::openmode mode);
		//~file_scriber();
		void write(const std::string &str) const;

		file_scriber &operator=(const file_scriber &src) {
			file = src.file;
			return *this;
		}

	private:
		sp_fstream file;
	};

} // namespace detail
} // namespace log
} // namespace blue_sky


#endif  // #ifndef BS_LOG_SCRIBERS_H_

