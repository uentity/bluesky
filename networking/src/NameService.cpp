#include "pch.h"

#include "networking/NameService.h"
#include "networking/ResourceManager.h"

using namespace std;
using namespace boost;

using namespace blue_sky;
using namespace blue_sky::networking;

#define PATH_SEPARATOR '/'

namespace {

	struct char_range
	{
		char * begin;
		char * end;
	};	

	bool skip_path_separator(const char ** l, const char **r)
	{
		while (**l == PATH_SEPARATOR)
			++(*l);

		while (**r == PATH_SEPARATOR)
			++(*r);

		return **l == 0 || **r == 0;
	}

	
	bool compare(const char **l, const char **r)
	{
		while (**l != PATH_SEPARATOR && **l != 0
			&& **r != PATH_SEPARATOR && **r != 0)
		{
			if (**l != **r)
				return false;
			else
			{
				++(*l);
				++(*r);
			}
		}
		return (**l == PATH_SEPARATOR || **l == 0) 
			&& (**r == PATH_SEPARATOR || **r == 0);
	}

	//returns length of matching fragment. Treats multiple consecutive slashes as one.
	//Example: find_difference("/1/2/3", "/1/22/4") => 3 (length of ("/1/"));
	size_t find_difference(const char * left, 
								 const char * right)
	{	
		const char * prefix = left;
		const char * l = left;
		const char * r = right;
		skip_path_separator(&l, &r);

		for (;;) {

			if (!compare(&l, &r))
				return prefix - left;
			prefix = l;		
			if (skip_path_separator(&l, &r))
				return l - left;
		}
	}

	template<typename IT>
	void advance_iterator(IT & it, const IT & end)
	{
		++it;
		while (it != end && it->length() == 0)
			++it;		
	}

	int get_common_prefix_length(const string & s1, const string & s2, string & suffix)
	{
		if (s2 == "/")
		{
			suffix = s1;
			return 1;
		}
		typedef boost::tokenizer<boost::char_separator<char> > 
			tokenizer;
		boost::char_separator<char> sep("/");
		tokenizer tk1(s1, sep);
		tokenizer tk2(s2, sep);

		tokenizer::iterator it1, it2;

		for (
				it1 = tk1.begin(), 
				it2 = tk2.begin()
			;
				it1 != tk1.end() && 
				it2 != tk2.end()
			;
				advance_iterator(it1, tk1.end()),
				advance_iterator(it2, tk2.end()))
		{
			if (*it1 != *it2)
				return -1;
		}

		if (it2 != tk2.end())
			return -1;

		suffix = "";
		while (it1 != tk1.end())
		{
			suffix += PATH_SEPARATOR;
			suffix += *it1;
			++it1;
		}
		BOOST_ASSERT(s1.length() > suffix.length());
		return s1.length() - suffix.length();
	}
}

void NameService::add(blue_sky::networking::ResourceManager *manager)
{
	mutex::scoped_lock lock(map_mutex_);	
	//cout << "NameService::add(" << manager->path_prefix() << ");" << std::endl;
	if (!map_.insert(make_pair(manager->path_prefix(), manager)).second)	
	{
		throw bs_exception(
			"NameService", 
			(std::string("Path prefix '") + manager->path_prefix() + "' already registered.").c_str()
			);
	}
}

void NameService::remove(const std::string &path_prefix)
{
	mutex::scoped_lock lock(map_mutex_);
	map_.erase(path_prefix);
}

void NameService::remove(blue_sky::networking::ResourceManager *manager)
{
	mutex::scoped_lock lock(map_mutex_);
	typedef map<string, ResourceManager *> TMap;	
	TMap::iterator it, end_it;
	for (it = map_.begin(), end_it = map_.end();
		it != end_it; ++it) 
	{
		if (it->second == manager) 
			break;
			
	}
	if (it != map_.end()) 
		map_.erase(it);
}

bool NameService::lookup(const std::string &name, ResourceManager ** rm, string & suffix) const
{
	mutex::scoped_lock lock(map_mutex_);

	typedef map<string, ResourceManager *>::const_iterator IT;	

	*rm = 0;
	suffix = "";

	int best_match = 0;

	//Неэффективный линейный поиск
	for (IT it = map_.begin(), end_it = map_.end();
		it != end_it; ++it)
	{
		int match = get_common_prefix_length(name, it->first, suffix);
		//size_t match = find_difference(name.c_str(), it->first.c_str());
		if (match > best_match)
		{
			best_match = match;
			*rm = it->second;
			//suffix = name.substr(match);
		}
	}		
	//cout << "Name Service: " << name << " =>> " << (*rm != 0 ? (*rm)->path_prefix() : " null ") << std::endl;
	return *rm != 0;
}

void NameService::dump(std::ostream &stream) const
{
	mutex::scoped_lock lock(map_mutex_);
	typedef map<string, ResourceManager *>::const_iterator IT;
	for (IT it = map_.begin(), end_it = map_.end(); it != end_it; ++it)
	{
		stream << "\"" << it->first << "\" => " 
			<< typeid(it->second).name() << "\n";
	}
}

std::ostream & blue_sky::networking::operator<<(std::ostream & stream, NameService const& service){
			service.dump(stream);
			return stream;
}