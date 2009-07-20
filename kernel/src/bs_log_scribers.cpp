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

	void cout_scriber::write(const std::string &str) const {
//#ifdef _DEBUG
    // TODO: miryanov
    static bool is_buffer_installed = false;
    if (!is_buffer_installed)
      {
        static char cout_buffer [2*4096] = {0};
        cout.rdbuf ()->pubsetbuf (cout_buffer, sizeof (cout_buffer));
        is_buffer_installed = true;
      }

    cout << str.c_str ();
//#endif
	}

	file_scriber::file_scriber(const std::string &filename, ios_base::openmode mode)
		: file(new fstream(filename.c_str(),mode))
	{}

	//file_scriber::~file_scriber() {
	//	file.lock()->close();
	//}

	void file_scriber::write(const std::string &str) const {
#ifdef _DEBUG
    // TODO: miryanov
		*(file.lock()) << str;
#endif
	}

} // namespace detail
} // namespace log 
} // namespace blue_sky

