#ifndef __Response_H__3F673BA5_04CB_4CBD_8016_958A225B5C09_
#define __Response_H__3F673BA5_04CB_4CBD_8016_958A225B5C09_

#include <iosfwd>

#include <string>

#include <networking/Context.h>

namespace blue_sky
{
	namespace networking
	{
		//Сообщение - ответ сервера
		class Response
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;			
		public:
			Response(ContextPtr context);
			int Status_code;
			std::string Reason_Phrase;
			std::string ETag;			
			time_t Last_Modified;
			sp_obj Body;

			std::ostream & write_to_stream(std::ostream &)const;
			std::istream & read_from_stream(std::istream &);
		};

		inline std::ostream & operator << (
			std::ostream & stream,
			const Response & message
		)
		{
			return message.write_to_stream(stream);
		}

		inline std::istream & operator >> (
			std::istream & stream,
			Response & message
		)
		{
			return message.read_from_stream(stream);
		}
	}
}

#endif //__Response_H__3F673BA5_04CB_4CBD_8016_958A225B5C09_