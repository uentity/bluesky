/**
 * \file bs_log_stream.h
 * \brief 
 * \author Sergey Miryanov
 * \date 07.07.2009
 * */
#ifndef BS_LOG_STREAM_H_
#define BS_LOG_STREAM_H_

#include "bs_common.h"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

namespace blue_sky {
namespace log {

    //there is error with smart_pointer access... I don't know why.
	class BS_API bs_stream { //: public bs_refcounter { //!!!!!!!!!!!!!!!!!!!!!!!
	public:
		virtual bool subscribe(bs_channel&);
		virtual bool unsubscribe(bs_channel&);
		virtual void write(const std::string&) const = 0; //{}//= 0;//{std::cout << "asdddd" << std::endl;}
		//virtual void dispose() const;

		virtual ~bs_stream() {}
	};

} // namespace log
} // namespace blue_sky


#endif // #ifndef BS_LOG_STREAM_H_

