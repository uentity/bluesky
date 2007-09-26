#ifndef __URI_H___14BB6DD1_B578_4975_B2B6_E041C9FA6AD0
#define __URI_H___14BB6DD1_B578_4975_B2B6_E041C9FA6AD0

#include <boost/smart_ptr.hpp>
#include <string>
#include <networking/lib.h>

namespace blue_sky
{
	namespace networking
	{
		/*
			Splits uri "protocol://host/path" into "protocol", "host" and "path".
			If protocol not specified (i.e. no "://") then entire uri is a path.
		*/
		class  BSN_API Uri
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;						
		public:
			Uri();
			Uri(const char*);
			Uri(std::string const&);			
			std::string const& protocol()const;
			std::string const& host()const;
			std::string const& path()const;
			std::string port()const;
			const std::string & str()const;
			const char * c_str()const;

			Uri append(const std::string & path)const;
		};
	}	
}

#endif //__URI_H___14BB6DD1_B578_4975_B2B6_E041C9FA6AD0