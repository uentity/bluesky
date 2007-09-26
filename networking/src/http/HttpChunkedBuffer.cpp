#include "pch.h"

#include "HttpChunkedBuffer.h"

using namespace std;

using namespace blue_sky::http_library;

namespace {

	bool match_CRLF(istream * stream)
	{
		if (stream->peek() == '\r') {
			stream->get();
			if (stream->peek() == '\n') {
				stream->get();
				return true;
			} else {
				stream->unget();
				return false;
			}		
		} else {
			return false;
		}
	}

}


HttpChunkedBuffer::HttpChunkedBuffer(istream * in, ostream * out, int chunk_size)
:
last_chunk_read_(false)
{	
	BUFF_SIZE = chunk_size;
	buff_ = new char[BUFF_SIZE];
	in_ = in;
	out_ = out;
	error = false;
	
	output_chunk_size_ = chunk_size;
	out_buffer_.resize(chunk_size);

	current_chunk_size_ = -1;

	fill_buffer();
	setp(&out_buffer_[0], &out_buffer_[0] + chunk_size);		
}

void HttpChunkedBuffer::fill_buffer()
{
	if (in_ == 0)
		return;
		
	if (current_chunk_size_ < 0 || current_chunk_left_ <= 0)
	{
		read_chunk_size();
		current_chunk_left_ = current_chunk_size_;
		if (current_chunk_size_ == 0)
		{
			last_chunk_read_ = true;
			return;
		}
	} 
			
	int bytes_to_read = BUFF_SIZE < current_chunk_left_ ? BUFF_SIZE : current_chunk_left_;
	in_->read(buff_, bytes_to_read);	
	error = in_->fail();
	setg(buff_, buff_, buff_ + bytes_to_read);
	current_chunk_left_ -= bytes_to_read;
	 
	if (current_chunk_left_ == 0)
	{
		if (!match_CRLF(&*in_))
			error = true;
	}
}

HttpChunkedBuffer::~HttpChunkedBuffer()
{
	delete[] buff_;
}

HttpChunkedBuffer::int_type HttpChunkedBuffer::uflow()
{
	if (in_->eof())
		return streambuf::traits_type::eof();
	if (error)
		return streambuf::traits_type::eof();	
	if (current_chunk_left_ <= 0)
	{
		read_chunk_size();
		if (current_chunk_size_ == 0)
		{
			last_chunk_read_ = true;
			return streambuf::traits_type::eof();
		}
	} 
	if (current_chunk_left_ > 0)
	{
		current_chunk_left_--;
		return in_->get();
	}else {
		return streambuf::traits_type::eof();
	}
}

streambuf::int_type HttpChunkedBuffer::underflow()
{
	if (in_->eof())
		return streambuf::traits_type::eof();
	if (error)
		return streambuf::traits_type::eof();
	fill_buffer();
	if (current_chunk_size_ == 0)
		return streambuf::traits_type::eof();
	return buff_[0];	
}

namespace
{
	template<typename IT>
	void write_chunk(ostream * out, IT begin, IT end, size_t size)
	{		
		*out << uppercase << hex << size << "\r\n" << nouppercase << dec;
		for (IT it = begin, end_it = end; it != end_it; ++it)
		{
			out->put(*it);
		}	
		*out << "\r\n";
}
}

HttpChunkedBuffer::int_type HttpChunkedBuffer::overflow(HttpChunkedBuffer::int_type ch)
{
	if (!out_->good())
		return traits_type::eof();
	typedef vector<traits_type::char_type>::iterator IT;
	write_chunk(out_, out_buffer_.begin(), out_buffer_.end(), out_buffer_.size());	
	out_->flush();	
	out_buffer_[0] = ch;
	setp(&out_buffer_[0], &out_buffer_[0] + 1, &out_buffer_[0] + output_chunk_size_);
	if (!out_->good())
		return traits_type::eof();
	return traits_type::not_eof(ch);	
}

int HttpChunkedBuffer::sync()
{
	size_t size = pptr() - pbase();
	if (size == 0)
		return 0;
	write_chunk(out_, pbase(), pptr(), size);
	out_->flush();
	setp(&out_buffer_[0], &out_buffer_[0] + output_chunk_size_);
	return out_->good() ? 0 : -1;
}

void HttpChunkedBuffer::write_last_chunk()
{
	sync();
	*out_ << "0\r\n\r\n";
	out_->flush();
}

namespace {
	bool is_hex(char c)
	{
		char c2 = toupper(c);
		return ('0' <= c && c <= '9') || ('A' <= c2 && c2 <= 'F');
}

}

namespace
{
	/*
		Reads size of next chunk in hex representation.
		Throws std::exception in following cases:
			- chunk size is not followed by CRLF
			- chunk size is too large. (more than 100 hexadecimal digits for now.)
	*/
	int read_chunk_size(istream * stream)
	{	
		int digits = 0;
		std::string str;
	
		while (!stream->eof() && is_hex(stream->peek())) {
			str.push_back(stream->get());
			++digits;
			if (digits > 100) {
				throw exception("Too large chunk.");
			}
		}
		while (!stream->eof() && stream->peek() != '\r') {
			stream->ignore(10, '\r');
		}
		if (!match_CRLF(stream))
		{
			ostrstream message;
			message << "Invalid chunk size: " << str << ". CRLF expected." << endl;
			throw std::exception(message.str());		
		} else {
			int result;			
			if (sscanf(str.c_str(), "%x", &result) != 1) {
				ostrstream message;
				message << "Invalid chunk size: " << str << endl;
				throw std::exception(message.str());					
			} else
				return result; 
		}	
	};

}

namespace
{

	void skip_trailer(std::istream * in_)
	{
		do 
		{		
			while (in_->get() != '\r'){}
		} while(in_->get() != '\n');
	}

}

void HttpChunkedBuffer::read_chunk_size()
{	
	if (last_chunk_read_)
		return;
	try {
		current_chunk_size_ = ::read_chunk_size(in_);	
		current_chunk_left_ = current_chunk_size_;
		if (current_chunk_size_ == 0)
			skip_trailer(in_);
	} catch (exception &) {
		error = true;		
	}
	pos_ = 0;
}

bool HttpChunkedBuffer::last_chunk_read()
{
	return last_chunk_read_;
}




