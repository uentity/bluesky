/**
 * \file bs_log_scribers.cpp
 * \brief
 * \author Sergey Miryanov
 * \date 07.07.2009
 * */
#include "bs_log_scribers.h"

using namespace std;

namespace blue_sky {
namespace log {
namespace detail {

#ifdef _DEBUG
#ifdef BS_LOG_USE_BUFFER
#undef BS_LOG_USE_BUFFER
#endif
#define BS_LOG_USE_BUFFER 0
#endif

#ifndef BS_LOG_USE_BUFFER
#define BS_LOG_USE_BUFFER 1
#endif

#ifndef BS_LOG_USE_BUFFER_FORCE
#define BS_LOG_USE_BUFFER_FORCE 0
#endif

#if BS_LOG_USE_BUFFER_FORCE
#define BS_LOG_USE_BUFFER 1
#endif

	void cout_scriber::write(const std::string &str) const {
#if BS_LOG_USE_BUFFER
    static bool is_buffer_installed = false;
    if (!is_buffer_installed)
      {
        static char cout_buffer [2*4096] = {0};
        cout.rdbuf ()->pubsetbuf (cout_buffer, sizeof (cout_buffer));
        is_buffer_installed = true;
      }
#endif

    cout << str;
	}

	file_scriber::file_scriber(const std::string &name, const std::string &filename, ios_base::openmode mode)
		: bs_stream (name), 
    file(new fstream(filename.c_str(),mode))
	{}

	//file_scriber::~file_scriber() {
	//	file.lock()->close();
	//}

	void file_scriber::write(const std::string &str) const {
		*(file) << str;
	}

} // namespace detail
} // namespace log 
} // namespace blue_sky

