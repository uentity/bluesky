#include "pch.h"
#include <networking/folder_listing.h>

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;

typedef std::vector<listing_entry_t> entries_t;

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive & ar, listing_entry_t & le, const unsigned int version)
{
    ar & le.name;
    ar & le.isFolder;
	ar & le.mtime;
}

} // namespace serialization
} // namespace boost

std::ostream & blue_sky::networking::operator <<(std::ostream & stream, const listing_entry_t & entry)
{
	if (entry.isFolder)
		stream << "[" << entry.name << "]";
	else
		stream << entry.name;
	return stream;
}

std::ostream & blue_sky::networking::operator<<(
	std::ostream& stream, 
	const folder_listing & listing)
{
	for (size_t i = 0; i < listing.size(); ++i)
	{
		stream << listing[i] << std::endl;
	}
	return stream;
}

class folder_listing::Impl
{
	friend class boost::serialization::access;
    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {		
		size_t size = fEntries.size();
		ar & size;
		for (size_t i = 0; i < fEntries.size(); ++i)
		{
			ar & fEntries.at(i);
		}	        
    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        size_t size;
		ar & size;
		fEntries.resize(size);
		for (size_t i = 0; i < size; ++i)
			ar & fEntries.at(i);
	}
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
	entries_t fEntries;
};

folder_listing::folder_listing(blue_sky::bs_type_ctor_param param)
: pimpl(new Impl())
{
	
}

folder_listing::~folder_listing()
{
}

void folder_listing::fill_from(const std::vector<listing_entry_t> &data)
{
	pimpl->fEntries.assign(data.begin(), data.end());
}

void folder_listing::add(const listing_entry_t & entry)
{
	pimpl->fEntries.push_back(entry);
}

size_t folder_listing::size()const
{
	return pimpl->fEntries.size();
}

const listing_entry_t & folder_listing::at(size_t idx)const
{
	return pimpl->fEntries.at(idx);
}

void folder_listing::serialize(std::ostream & stream) const
{	
	boost::archive::text_oarchive ar(stream);
	const Impl * impl = pimpl.get();
	ar << *impl;
}

void folder_listing::deserialize(std::istream & stream)
{
	boost::archive::text_iarchive ar(stream);
	ar >> *pimpl;	
}


namespace blue_sky
{
	namespace networking
	{
		BLUE_SKY_TYPE_STD_CREATE(folder_listing);
		BLUE_SKY_TYPE_IMPL_NOCOPY_SHORT(folder_listing, objbase, "Represent fodler listing");
	}
}