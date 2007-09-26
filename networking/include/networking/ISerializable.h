#ifndef ISERIALIZABLE_H_10BD8B0C_4CE7_4414_9952_85C05C014007
#define ISERIALIZABLE_H_10BD8B0C_4CE7_4414_9952_85C05C014007

#include <iosfwd>

namespace blue_sky
{
namespace networking
{
	class ISerializable
	{
	public:
		virtual void serialize(std::ostream &)const = 0;
		virtual void deserialize(std::istream &) = 0;
	};
}
}

#endif