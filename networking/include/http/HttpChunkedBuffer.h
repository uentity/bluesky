#ifndef __HTTPCHUNKEDBUFFER_H__C2EBBA5C_12C1_4F2A_ABDF_35AACBE7A410_
#define __HTTPCHUNKEDBUFFER_H__C2EBBA5C_12C1_4F2A_ABDF_35AACBE7A410_

#include <iosfwd>
#include <streambuf>
#include <vector>

#include <boost/iostreams/char_traits.hpp> 
#include <boost/iostreams/concepts.hpp>  
#include <boost/iostreams/operations.hpp> 


namespace blue_sky
{
	namespace http_library
	{		

		class HttpChunkedBuffer : public std::streambuf
		{
			bool last_chunk_read_;
		public:
			typedef std::streambuf::int_type int_type;
			typedef std::streambuf::traits_type traits_type;
		private:
			int BUFF_SIZE;
			std::istream * in_;
			std::ostream * out_;
			char * buff_;
			std::vector<traits_type::char_type> out_buffer_;
			
			int output_chunk_size_;
			int pos_;

			int current_chunk_size_;
			int current_chunk_left_;
			bool error;
		protected:			 
			virtual int_type uflow( );
			virtual int_type underflow( );
			virtual int_type overflow(int_type);
			virtual int sync();
		public:
			HttpChunkedBuffer(std::istream * in, std::ostream * out, int chunk_size);
			virtual ~HttpChunkedBuffer();
			void write_last_chunk();
			bool last_chunk_read();
		private:
			void read_chunk_size();
			char read();		
			void fill_buffer();
		};
	}
}

#endif //__HTTPCHUNKEDBUFFER_H__C2EBBA5C_12C1_4F2A_ABDF_35AACBE7A410_