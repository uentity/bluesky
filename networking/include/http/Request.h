#ifndef __Request_H__C59A68B8_4626_41C7_ACB8_B8A8EE97DBCA_
#define __Request_H__C59A68B8_4626_41C7_ACB8_B8A8EE97DBCA_

#include <iosfwd>
#include <string>

#include <bs_fwd.h>

#include <networking/Uri.h>

namespace blue_sky
{
	namespace networking
	{
		//Сообщение - клиентский запрос 
		class Request
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;			
		public:				
			Request(ContextPtr context);

			enum {M_GET, M_PUT, M_POST, M_DELETE, M_LOCK, M_UNLOCK} method;
			Uri uri;
			std::string If_Match;					
			sp_obj Body;

			std::ostream & write_to_stream(std::ostream &)const;
			std::istream & read_from_stream(std::istream &);
		};

		inline std::ostream & operator << (
			std::ostream & stream, 
			const Request & message
		)
		{
			return message.write_to_stream(stream);
		}

		inline std::istream & operator >> (
			std::istream & stream, 
			Request & message
		)
		{
			return message.read_from_stream(stream);
		}

	}
}

#endif //__Request_H__C59A68B8_4626_41C7_ACB8_B8A8EE97DBCA_