#ifndef __CONNECTION_H__DE2440CA_6E90_4C61_9B31_E0B3DAA88056_
#define __CONNECTION_H__DE2440CA_6E90_4C61_9B31_E0B3DAA88056_

#include <string>
#include <boost/smart_ptr.hpp>

#include <networking/Context.h>


namespace blue_sky {
namespace networking {

class Request;
class Response;

class Connection;

typedef boost::shared_ptr<Connection> ConnectionPtr;


// - Отправляет сообщения (LocalConnection, HttpConnection)
class Connection
{	
	std::string uri_;	
	Connection(Connection const&);
public:
	Connection(std::string const& uri);
	virtual ~Connection();

	const	std::string & uri();

	virtual void send(
		const Request & request, 
		Response & response
	) = 0;	

	virtual void close() = 0;

public:
	static ConnectionPtr connect(
		ContextPtr context, 
		std::string uri
	);
};

}}

#endif //__CONNECTION_H__DE2440CA_6E90_4C61_9B31_E0B3DAA88056_