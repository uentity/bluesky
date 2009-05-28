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
# custom_vars contains two predefined build variables, 'debug' with default
# value '1', and 'release' with default value '0'. These variables controls
# whether your script should build a debug and/or release version of code
# (with debug enabled by default). custom_env is initialized as emty scons
# environment with associated custom_vars. Note that default values for build
# variables will be read from file scons_vars.custom, if it exists. This
# allows you to omit them on command line without having to edit 
# corresponding sconscripts.
#
# When all initialization is done, three variables ss_tree, custom_vars and
# custom_env are shared via scons's Export() function in order to be
# accessible from within another build scripts.
#
# The key point of underlying build framework is custom user's processing
# script scons_env.custom, which is invoked (if exists) before all other
# sconscripts. The main task of scons_env.custom is to modify custom_env,
# custom_vars and ss_tree according to specific user-related environment.
# Such environment includes paths to libraries, machine-related compiler
# switches, etc. Newly introduced build variables may be placed to
# custom_vars and all build scripts should be added to ss_tree. By default,
# only BlueSky kernel and bspy_loader is build, so if you want to build some
# plugins, just append corresponding sconscripts to ss_tree list.
# Note, that ss_tree is ORDER-SENSITIVE, so be shure to include all
# dependencies _before_ dependent script.
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
# easily adoptable by new developers.
#
# So, basically adjusting the build process to your environment includes the
# following steps.
# 1) Create script scons_vars.custom, where you can place custom defaul values
# for any build variables (see scons documentation on this).
# 2) Create script scons_env.custom.
# 2.1) Modify custom_env according to your environment.
# 2.2) Place newly added build variables (if any) to custom_vars. Update
# custom_env with added variables via scons_vars.Update(scons_env). Note, that
# if your build script depends on the value of specific build variable, then
# you should create such variable inside dependent script and not in the
# scons_env.custom, because if by any cahnce scons_env.custom will be
# absent, your build script would fail with error (unless you check for
# existence of needed variable before checking it's value).
# 2.3) Extend list of build scripts by appending new scripts to ss_tree
# variable.
# 2.4) Export changes done via Export('custom_vars', 'custom_env', 'ss_tree')
# Although it seems that exporting custom_vars and custom_env only once in the
# beginning of SConstruct is enough to track all changes, you MUST explicitly
# export ss_tree, otherwise main SConstruct won't see your changes after
# exiting from scons_env.custom.
#
# You can place additional checks, tunings, and whatever you like inside
# scons_env.custom unless it satisfies above conditions. You should always
# prefer invoking scons_env.custom from your plugin's build script rather than
# inlining all custom tunings. In general you shouldn't rely on
# existence of scons_env.custom and your script should provide minimal build
# without it. But possible later forcing the user to have such script would be a
# good tradeoff between reqirements and great stability/portablility/support
# of build system, which concentrate all user-related setup in one place.
#
# En example of scons.env.custom, called scons_env.custom.example is shipped
# with BlueSky kernel source bundle, so you can use it as a template for your
# own code.
#
# Inside your plugin's sconscript you would normally do
# Import('*') to import all stuff exported earlier, includin custom_env,
# then create your own build environment by cloning custom_env:
# my_build_env = custom_env.Clone();
# After that you'll have all specific settings inside my_build_env. As a
# rule, you must check build variables 'debug' and 'release' in order to
# provide corresponding build type. You script should be able to build both
# when scons is invoked like 'scons debug=1 release=1'. See kernel's
# SConscript on exmple of how to do it.
#
# Update 09.04.2009: There is a possibility to change values of default build
# variables, that are hardcoded inside build sconscripts at the moment of
# addition to custom_vars. The approach relies on scons's future allowing to read
# such values from a custom user's python script. That script by convention
# should be called scons_vars.custom in BlueSky. If it exists in the top
# source directory (from where SConstruct is run), then scons_vars on creation
# is passed with paramter 'scons_vars.custom', which forces to read default
# values from given file. The scons_vars.custom is processed like python
# script and has a very simple format:
# var1 = value
# var2 = value

import os, os.path;

def custom_proc_call() :
	# process custom settings
	if os.path.exists('scons_env.custom') :
		SConscript('scons_env.custom');

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
custom_vars.Add('debug', 'Set to 1 to build debug', '1');
custom_vars.Add('release', 'Set to 1 to build release', '0');
# decide whether we consider dependencies or not - needed to rebuild given target ONLY
custom_vars.Add('nodeps', 'Set to 1 to build ignoring dependencies', '0');
# option that controls whether to perform install step after build
custom_vars.Add('install', 'Set to 1 to install after build', '0');
# if user pointed to make install --- add path prefix variables describing where to install kernel and plugins
#if custom_env['install'] == '1' :
	# accept any path setting because Install builder will create them automatically
#	custom_vars.Add('prefix', 'Point where to install BlueSky kernel', 'lib');
custom_vars.Add(PathVariable('prefix', 'Point where to install BlueSky kernel', Dir('#lib').get_abspath(),
	PathVariable.PathAccept));
#	custom_vars.Add('plugins_prefix', 'Point where to install BlueSky plugins', '$prefix/plugins');
custom_vars.Add(PathVariable('plugins_prefix', 'Point where to install BlueSky plugins', '$prefix/plugins',
	PathVariable.PathAccept));
#	# add new vars to custom_env
#	custom_vars.Update(custom_env);
#	# if paths doesn't exists - create them
#	prefix = custom_env.subst('$prefix');
#	plugins_prefix = custom_env.subst('$plugins_prefix');
#	if not os.path.exists(prefix) :
#		os.mkdir(prefix);
#	if not os.path.exists(plugins_prefix) :
#		os.mkdir(plugins_prefix);

# variable that specifies python version installed in the system
custom_vars.Add('python_name', 'Put full Python interpreter name with version here, ex. python2.5', 'python2.5');
# debug compile flags
#custom_vars.Add('ccflags_dbg', 'Specify compiler flags for debug build', '-O0 -ggdb3');
# release compile flags
#custom_vars.Add('ccflags_rel', 'Specify compiler flags for release build', '-O3');

# create custom environment
custom_env = Environment(variables = custom_vars);

# export created variables
Export('ss_tree', 'custom_vars', 'custom_env');

# setup commonly used names
dbg_dir = 'debug';
rel_dir = 'release';
build_dir = '#build';
exe_dir = '#exe';
Export('dbg_dir', 'rel_dir', 'build_dir', 'exe_dir');

# determine what should we build
build_kinds = [];
if custom_env['debug'] == '1' :
	build_kinds += ['debug'];
if custom_env['release'] == '1' :
	build_kinds += ['release'];
Export('build_kinds');

# initialization stage is for correcting invariants, such as ss_list, etc
build_kind = 'init';
Export('build_kind');
# custom script call
custom_proc_call();
# import full sconscripts list
Import('ss_tree');
# just parse all scripts to let them make initialization
#[SConscript(x) for x in ss_tree];

# start global build cycle for every build kind
for i in range(len(build_kinds)) :
	build_kind = build_kinds[i];
	# inform everyone what are we building now
	Export('build_kind');
	# format build path
	tar_build_dir = os.path.join(build_dir, build_kind);
	# format exe path
	tar_exe_dir = os.path.join(exe_dir, build_kind);
	Export('tar_build_dir', 'tar_exe_dir');
	# obtain custom environment
	custom_proc_call();
	# add exe path to libraries search paths
	custom_env.AppendUnique(LIBPATH = [tar_exe_dir]);
	if build_kind == 'debug' :
		custom_env.AppendUnique(CPPDEFINES = ['_DEBUG']);
	build_root = tar_build_dir;

	# parse scons files
	for j in range(len(ss_tree)) :
		# build in separate dir
		tar_build_dir = os.path.join(build_root, os.path.dirname(ss_tree[j]));
		inst_path = SConscript(ss_tree[j], variant_dir = tar_build_dir, duplicate = 0);
		# install to specified location
		#if not inst_path is None :
		#Install(tar_build_dir, os.path.join(tar_exe_dir, inst_path));
		#if scons_env['install'] == '1' :

# debug
#print 'sconscripts to build';
#print(ss.items());
#print 'build types:';
#print(build_targets);

#print ss_tree;
#Help(custom_vars.GenerateHelpText(custom_env));

# generate help text
Help(custom_vars.GenerateHelpText(custom_env));

