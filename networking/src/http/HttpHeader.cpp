#include "pch.h"
#include <HttpHeader.h>
//#include <HttpLibraryException.h>


using namespace blue_sky;
using namespace blue_sky::http_library;

void HttpHeaderCollection::add(const HttpHeader & header)
{
	add(header.name, header.value);	
}

void HttpHeaderCollection::add(const string & name, const string & value)
{
	if(!map_.insert(make_pair(name, value)).second)
	{
		throw std::exception(
			(string("HttpHeaderCollection::add. Header named '") + name + "' already exist.").c_str());
	}
}

const string & HttpHeaderCollection::get_value(const string & name)const
{
	internal_collection_::const_iterator it = map_.find(name);
	if (it != map_.end())
		return it->second;
	else
		throw std::exception((string("HttpHeaderCollection::get_value. No header named '") + name + "'.").c_str()) ;
}

bool blue_sky::http_library::HttpHeaderCollection::contains(const string & name)const
{	
	internal_collection_::const_iterator it = map_.find(name);
	return it != map_.end();
}