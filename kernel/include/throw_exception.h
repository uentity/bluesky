/** 
 * \file throw_exception.h
 * \brief for throw exception from bs_bos_core methods
 * \author Sergey Miryanov
 * \date 27.04.2009
 * */
#ifndef BS_THROW_EXCEPTION_62b5cf89_6093_4fd3_91ae_e674a9b9ae6c_H_
#define BS_THROW_EXCEPTION_62b5cf89_6093_4fd3_91ae_e674a9b9ae6c_H_

#include <boost/current_function.hpp>
#include "bs_exception.h"
#include <string>

namespace blue_sky {

#ifdef _DEBUG
#define bs_throw_exception(msg) \
  throw bs_exception (std::string (__FILE__) + ":" + boost::lexical_cast <std::string> (__LINE__) + "\n[" + BOOST_CURRENT_FUNCTION + "]", msg);
#else
#define bs_throw_exception(msg) \
  throw bs_exception (std::string (__FILE__) + ":" + boost::lexical_cast <std::string> (__LINE__), msg);
#endif

}



#endif // #ifndef BS_THROW_EXCEPTION_62b5cf89_6093_4fd3_91ae_e674a9b9ae6c_H_
