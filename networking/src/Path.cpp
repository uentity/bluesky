#include "pch.h"
#include <networking/Path.h>

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;

class Path::Impl
{	
public:
	PathQueue fNames;
	mutable std::string fPath;
};

Path::Path()
: pimpl(new Impl())
{
	split("");
	join();
}

Path::Path(const char * path)
: pimpl(new Impl())
{
	split(string(path));
	join();
}

Path::Path(const string & path)
: pimpl(new Impl())
{
	split(path);
	join();
}

Path::Path(Path::const_iterator & begin, Path::const_iterator & end)
: pimpl(new Impl())
{
	for (Path::const_iterator it = begin; it != end; ++it)
	{
		pimpl->fNames.push_back(*it);
	}
	join();
}

void Path::split(const std::string & name)
{
	pimpl->fNames.clear();
	char_separator<char> sep("/");
	typedef boost::tokenizer<boost::char_separator<char> > 
		tokenizer;
	tokenizer tok(name, sep);
	for (tokenizer::iterator it = tok.begin(), end_it = tok.end();
		it != end_it;
		++it)
	{
		pimpl->fNames.push_back(*it);
	}
}

void Path::join()
{
	pimpl->fPath = "";
	for (const_iterator it = begin(),
		end_it = end();
		it != end_it;
	++it)
	{
		if (pimpl->fPath.length() > 0)
			pimpl->fPath += "/";
		pimpl->fPath += *it;
	}
}



const char* Path::c_str()const
{
	return str().c_str();
}

const string & Path::str()const
{
	pimpl->fPath = "";
	for (PathQueue::const_iterator it = pimpl->fNames.begin(),
		end_it = pimpl->fNames.end();
		it != end_it;
	++it)
	{
		if (pimpl->fPath.length() > 0)
			pimpl->fPath += "/";
		pimpl->fPath += *it;
	}

	return pimpl->fPath;
}

Path Path::up()const
{
	if (pimpl->fNames.size() == 0)
		return Path();

	Path result(begin(), end());
	result.pimpl->fNames.pop_back();
	return result;
}

Path Path::down(const std::string & name)const
{
	return down(Path(name));	
}

Path Path::down(const char *name)const
{
	return down(Path(name));
}

Path Path::down(const Path & path)const
{
	Path result;

	for (const_iterator it = begin(),
		end_it = end();
		it != end_it;
		++it)
		{
			result.pimpl->fNames.push_back(*it);
		}

	for (const_iterator it = path.begin(),
		end_it = path.end();
		it != end_it;
		++it)
		{
			result.pimpl->fNames.push_back(*it);
		}
	return result;
}

Path::iterator Path::begin()
{
	return pimpl->fNames.begin();
}

Path::const_iterator Path::begin()const
{
	return pimpl->fNames.begin();
}

Path::iterator Path::end()
{
	return pimpl->fNames.end();
}

Path::const_iterator Path::end()const
{
	return pimpl->fNames.end();
}

Path::reverse_iterator Path::rbegin()
{
	return pimpl->fNames.rbegin();
}

Path::reverse_iterator Path::rend()
{
	return pimpl->fNames.rend();
}

bool Path::subfolderof(const blue_sky::networking::Path &p) const
{
	for (const_iterator it1 = p.begin(), end_it1 = p.end(),
			it2 = begin(), end_it2 = end();	;
			++it1, ++it2)
	{
		if (it1 == end_it1 && it2 != end_it2)
			return true;
		if (it2 == end_it2)
			return false;
		if (*it1 != *it2)
		{
			return false;
		}
	}			
}

bool Path::imediatesubfolderof(const blue_sky::networking::Path &p) const
{
	for (const_iterator it1 = p.begin(), end_it1 = p.end(),
		it2 = begin(), end_it2 = end();	;
		++it1, ++it2)
	{
		if (it1 == end_it1 && it2 != end_it2)
		{
			return (++it2) == end_it2;
		}
		if (it2 == end_it2)
			return false;
		if (*it1 != *it2)
		{
			return false;
		}
	}			
}

bool Path::equalto(const blue_sky::networking::Path &p) const
{
	for (const_iterator it1 = p.begin(), end_it1 = p.end(),
		it2 = begin(), end_it2 = end();	;
		++it1, ++it2)
	{
		if (it1 == end_it1 && it2 == end_it2)
		{
			return true;
		}
		if (it1 == end_it1 || it2 == end_it2)
			return false;
		if (*it1 != *it2)
		{
			return false;
		}
	}			
}

Path Path::subtract(const blue_sky::networking::Path &p) const
{
	const_iterator it1 = begin();
	const_iterator end_it1 = end();	
	const_iterator it2 = p.begin();
	const_iterator end_it2 = p.end();	
	
	while (it2 != end_it2)
	{
		if (it1 == end_it1)
			return Path();
		++it1;
		++it2;
	}
	Path result;
	while (it1 != end_it1)
	{
		result.pimpl->fNames.push_back(*it1);
		++it1;
	}
	return result;
}