# This is a main build script for BlueSky integration platform.
#
# It's rather simple, but designed to be flexible enough to provide building
# framework for BlueSky kernel and various plugins.
#
# Main script does not more than invoking SConscript for all files, enumerated
# in ss_tree list, plus some additional steps. These steps include creation of
# custom environment custom_env, associated custom build variables set
# custom_vars and invoking user-defined script scons_env.custom.
#
# custom_vars contains some predefined build variables, ex. 'debug' with default
# value '1', and 'release' with default value '0'. These variables controls
# whether your script should build a debug and/or release version of code
# (with debug enabled by default). custom_env is initialized as emty scons
# environment with associated custom_vars. Note that default values for build
# variables will be read from file scons_vars.custom, if it exists in the top
# BlueSky directory from where SConstruct is run (scons_vars on creation
# is passed with paramter 'scons_vars.custom', which forces to read default
# values from given file). The scons_vars.custom is processed like python
# script and has a very simple format:
# var1 = value1
# var2 = value2
# This allows you to omit var1=value1 and var2=value2 on command line without 
# having to edit corresponding sconscripts.
#
# Common widely used approach is to place all specific settings inside OS user
# environment which is shared between all running programs. However, changing
# OS environment is not that easy, as editing simple Python script,
# containing all you need in one place. Also in latter case you have your
# settings applied immediately, upon next build start. Concentrating settings
# inside scons_env.custom makes build process independent of OS environment and
# more predictable. Of cause, if you want you can make use of your OS
# environment by passing needed values to custom_env. But general advise is
# to do all setup explicitly in scons_env.custom, making it self-explanatory and
# easily replaceable by new developers.
#
# When all initialization is done, variables such as ss_tree, custom_vars,
# custom_env, etc are shared via scons's Export() function in order to be
# accessible from within another build scripts.
#
# The key point of underlying build framework is custom user's processing
# script scons_env.custom, which is invoked (if exists) before all
# sconscripts for every build type. By default SConstruct supports two build
# types --- debug and release (and corresponding build variables), but this
# list can be extended during special 'init' stage.
#
# Build types list is stored in shared variable build_kinds. Current build
# type contained in variable build_kind. There is a special build kind, called
# 'init'. During init only scons_env.custom is parsed. The main purpose of
# this stage is to extend ss_tree list with plugins you wish to build and do
# other custom global setup.
#
# The main task of scons_env.custom is to modify custom_env
# and ss_tree according to specific user-related environment.
# Such environment includes paths to libraries, machine-related compiler
# switches, etc. All build scripts should be added to ss_tree. By default,
# only BlueSky kernel and bspy_loader is build, so if you want to build some
# plugins, just append corresponding sconscripts to ss_tree list.
# Note, that ss_tree is ORDER-SENSITIVE, so be shure to include all
# dependencies _before_ dependent script.
#
# Also after init stage, scons_env_custom is called before all other
# sconscripts in order to adjust build environment to specific build type. You
# can set proper compiler flags, etc depending on required assembly. Here, 3
# more shared variables are available to all sconscripts, including
# scons_env.custom --- tar_build_dir (root dir for object files, defaults to
# build/$build_kind), tar_exe_dir (root dir where all target libs should be
# placed after build, defaults to exe/$build_kind) and tar_exe_plugins
# (where plugin libs should be placed, defaults to $tar_exe_dir/plugins). This
# varables can be altered in scons_env.custom before the build starts.
#
# So, basically adjusting the build process to your environment includes the
# following steps.
# 1) Create script scons_vars.custom, where you can place custom defaul values
# for any build variables (see scons documentation on this).
# 2) Create script scons_env.custom.
# 2.1) Extend list of build scripts by appending new scripts to ss_tree
# variable during init build kind. Do other global things here.
# 2.2) Modify custom_env according to your environment for every specific
# build kind.
# 2.3) Export changes done via Export([varaibales_list])
# Although it seems that exporting custom_vars and custom_env only once in the
# beginning of SConstruct is enough to track all changes, you MUST explicitly
# export ss_tree, otherwise main SConstruct won't see your changes after
# exiting from scons_env.custom.
# 2.4 You can place additional checks, tunings, and whatever you like.
#
# Please, follow basic rules described below.
# 1) In general you shouldn't rely on existence of scons_env.custom and your 
# targets should build without it.
# 2) Place all build variables that your script depends on inplace, i.e. right
# before use and not in the scons_env.custom. User can easily change the
# default value of your variable via own scons_vars.custom and unexpected
# scons_env.custom won't break the build.
# 3) DO NOT replace custom_env with Environment() call in scons_env.custom.
# Use Import('*') and scons_env.Replace instead.
#
# En example of scons.env.custom, called scons_env.custom.example is shipped
# with BlueSky kernel source bundle, so you can use it as a template for your
# own code.
#
# Inside your plugin's sconscript you would normally do
# Import('*') to import all stuff exported earlier, includin custom_env,
# then create your own build environment by cloning custom_env:
# my_build_env = custom_env.Clone();

import os, os.path, glob;

# list of all sconscript files
# ORDER-SENSITIVE!
# add your sconscript only AFTER all dependent
ss_tree = [
		'kernel/SConscript',
		'python/bspy_loader/SConscript'
		];

# create custom variables telling whether to build debug and/or release
custom_vars = 0;
if os.path.exists('scons_vars.custom') :
	custom_vars = Variables(File('#scons_vars.custom').get_abspath());
else :
	custom_vars = Variables();

# what build kinds do we support?
def_build_kinds = ['debug', 'release'];
custom_vars.Add(ListVariable('build_kinds', 'List of supported build kinds', def_build_kinds[0], def_build_kinds));

# append some useful variables by default
#custom_vars.Add('nodeps', 'Set to 1 to build ignoring dependencies', '0');
custom_vars.Add(EnumVariable('deps', """Select how to track dependencies:
	auto - rely on SCons internal machinery,
	explicit - force using Depends(),
	off - build fully ignoring dependencies""",
	'auto', allowed_values = ['auto', 'explicit', 'off']));
custom_vars.Add('install', 'Set to 1 to install after build', '0');

# if user pointed to make install --- add path prefix variables describing where to install kernel and plugins
custom_vars.Add(PathVariable('prefix', 'Point where to install BlueSky kernel', Dir('#lib').get_abspath(),
	PathVariable.PathAccept));
custom_vars.Add(PathVariable('plugins_prefix', 'Point where to install BlueSky plugins', '$prefix/plugins',
	PathVariable.PathAccept));

custom_vars.Add('python_name', 'Put full Python interpreter name with version here, ex. python2.5', 'python2.5');
# add variable to decide whether to build with python support
custom_vars.Add('py', 'Set to 1 to build with Python support', '0');

custom_vars.Add(BoolVariable('auto_find_ss', 'Turn on automatic SConscripts search?', 0));

# search for platform-oriented scripts
platform_ss = glob.glob('scons_platform.*');
pnames = [''];
for p in platform_ss :
	ext = os.path.splitext(p)[1][1:];
	if len(ext) > 0 : pnames.append(ext);
custom_vars.Add(EnumVariable('platform', 'Specify the platform to build for', '', allowed_values = pnames));

custom_vars.Add(PathVariable('custom_script', 'Specify filename of custom build environment processing script',
	'scons_env.custom', PathVariable.PathAccept));

# create custom environment
custom_env = Environment(variables = custom_vars);
Export('custom_vars', 'custom_env');

def custom_proc_call() :
	# process custom settings
	if os.path.exists(custom_env['custom_script']) :
		SConscript(custom_env['custom_script']);

# extract build kinds specified by user
build_kinds = custom_env['build_kinds'];
Export('build_kinds');

# auto serach for SConscripts
def ss_search(root_dir, ss_prefix, ss_list) :
	nodes = os.listdir(root_dir);
	for node in nodes :
		abs_node = os.path.join(root_dir, node);
		if not os.path.isdir(abs_node) : continue;
#		print 'Travel path ', abs_node;
		ss_glob = glob.glob(os.path.join(abs_node, '[Ss][Cc]onscript'));
		if len(ss_glob) > 0 :
			ss = os.path.join(ss_prefix, node, os.path.basename(ss_glob[0]));
#			print 'Found ', ss;
			if ss not in ss_list : ss_list.append(ss);
		else :
			ss_search(abs_node, os.path.join(ss_prefix, node), ss_list);

if custom_env['auto_find_ss'] :
#	print "Auto-search on";
	root_dir = Dir('#').get_abspath();
#	print 'root_dir = ', root_dir;
	ss_list = [];
	ss_search(root_dir, '', ss_tree);
#	print ss_tree;
Export('ss_tree');

# setup commonly used names
plugin_dir = 'plugins';
build_dir = '#build';
exe_dir = '#exe';
Export('build_dir', 'exe_dir', 'plugin_dir');

# import some useful tools
SConscript('scons_tools');

# initialization stage is for correcting invariants, such as ss_list, etc
build_kind = 'init';
Export('build_kind');
# custom script call
custom_proc_call();
Import('*');

# dump ss_tree
print 'Processing build scripts:';
print '[';
for x in ss_tree :
	print x;
print']';

# checkers
if not custom_env.GetOption('clean') and not custom_env.GetOption('help') :
	conf = Configure(custom_env, custom_tests = { 'CheckBoost' : CheckBoost });
	if not conf.CheckCXX() :
		print('!! Your compiler and/or environment is not correctly configured.');
		Exit(0);
	CheckBoost(conf);
	CheckLoki(conf);
	env = conf.Finish();

# save default custom_env
custom_env_def = custom_env.Clone();

#debug
def compare_env(env1, env2) :
	print 'env1: ', env1;
	print 'env2: ', env2;
	d1 = env1.Dictionary(); d2 = env2.Dictionary();
	i = 0;
	for k in d1.keys() :
		if d1[k] != d2[k] :
			print 'env1[', k, '] = ', d1[k];
			print 'env2[', k, '] = ', d2[k];
			++i;
	if i == 0 : print 'Dictionary match!';
	print '';

# start global build cycle for every build kind
platform = custom_env['platform'];
for i in range(len(build_kinds)) :
	# inform everyone what are we building now
	build_kind = build_kinds[i];
	Export('build_kind');
	# format root build and exe paths
	tar_build_dir = os.path.join(build_dir, build_kind);
	tar_exe_dir = os.path.join(exe_dir, build_kind);
	# where plugin libs are expected to be after build?
	tar_exe_plugin_dir = os.path.join(tar_exe_dir, plugin_dir);
	Export('tar_build_dir', 'tar_exe_dir', 'tar_exe_plugin_dir');

	# reset custom_env to default values
	custom_env = custom_env_def.Clone();
	Export('custom_env');
	# invoke tuning scripts
	if len(platform) > 0 :
		SConscript('scons_platform.' + platform);
	custom_proc_call();
	Import('*');

	# add exe path to libraries search paths
	custom_env.AppendUnique(LIBPATH = [tar_exe_dir, tar_exe_plugin_dir]);
	if build_kind == 'debug' :
		custom_env.AppendUnique(CPPDEFINES = ['_DEBUG']);
	elif (build_kind == 'release') :
		custom_env.AppendUnique(CPPDEFINES = ['NDEBUG']);

	Export('custom_env');
	
	# parse scons files
	build_root = tar_build_dir;
	for j in range(len(ss_tree)) :
		# build in separate dir
		tar_build_dir = os.path.join(build_root, os.path.dirname(ss_tree[j]));
		inst_path = SConscript(ss_tree[j], variant_dir = tar_build_dir, duplicate = 0);
		# install to specified location
		#if not inst_path is None :
		#Install(tar_build_dir, os.path.join(tar_exe_dir, inst_path));
		#if scons_env['install'] == '1' :
	
	# Update template env with possibly cahnged build variables
	custom_vars.Update(custom_env_def);

# generate help text
Help(custom_vars.GenerateHelpText(custom_env));

