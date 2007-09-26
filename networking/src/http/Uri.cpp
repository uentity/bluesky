#include "pch.h"

#include "networking/uri.h"

using namespace std;
using namespace blue_sky;
using namespace blue_sky::networking;

class Uri::Impl
{
public:
	std::string uri_;
	std::string protocol_;
	std::string host_;
	std::string path_;
	std::string port_;

	void init(const string & uri)
	{
		uri_ = uri;
		string proto_sep("://");
		string::size_type pos = uri.find(proto_sep); // "protocol|>pos<|://host:80/path"
		if (pos == string::npos)
		{
			protocol_ = "";
			host_ = "";
			path_ = uri;
		} else {
			protocol_ = uri.substr(0, pos);
		string::size_type pos2 = pos + proto_sep.length(); // "protocol://|>pos2<|host:80/path"
		string::size_type port_pos = uri.find(":", pos2); // "protocol://host|>port_pos<|:80/path"
		string::size_type pos3 = uri.find("/", pos2); // "protocol://host:80|>pos3<|/path"

		if (port_pos == string::npos || (pos3 != string::npos && pos3 < port_pos))
		{	
			port_ = -1;
			if (pos3 == string.npos)
			{
				host_ = uri.substr(pos2);
				path_ = "";
			} else {
				host_ = uri.substr(pos2, pos3 - pos2);
				path_ = uri.substr(pos3 + 1);
			}
		} else {
			if (pos3 == string.npos)
			{
				host_ = uri.substr(pos2, port_pos - pos2);
				port_ = (uri.substr(port_pos + 1));
				path_ = "";
			} else {
				host_ = uri.substr(pos2, port_pos - pos2);
				port_ = (uri.substr(port_pos + 1, pos3 - port_pos - 1));
				path_ = uri.substr(pos3 + 1);
			}
		}		
	}
	}
};

Uri::Uri()
:pimpl(new Impl())
{

}

Uri::Uri(string const& uri)
:pimpl(new Impl())
{
	pimpl->init(uri);
}

Uri::Uri(const char* uri)
:pimpl(new Impl())
{
	pimpl->init(uri);
}


string const& Uri::protocol()const
{
	return pimpl->protocol_;
}

string const& Uri::host()const
{
	return pimpl->host_;
}

string const& Uri::path()const
{
	return pimpl->path_;
}

string Uri::port()const
{
	return pimpl->port_;
}

const string & Uri::str()const
{
	return pimpl->uri_;
}

const char * Uri::c_str()const
{
	return pimpl->uri_.c_str();
}

Uri Uri::append(const std::string & path)const
{
	return Uri(pimpl->uri_ + "/" + path);
}
