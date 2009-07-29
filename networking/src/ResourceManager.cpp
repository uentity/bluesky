#include "pch.h"
#include "networking/ResourceManager.h"
#include "networking/NameService.h"


using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;

ResourceManager::ResourceManager(blue_sky::networking::NameService * name_service, const std::string &path_prefix)
: path_prefix_(path_prefix)
{
	name_service_ = name_service;
	name_service_->add(this);
}

ResourceManager::~ResourceManager()
{
	name_service_->remove(this);
}

std::string const& ResourceManager::path_prefix()
{
	return path_prefix_;
}

