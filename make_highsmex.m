function make_highsmex(highs_ver, mexSrcFiles)
% This function builds the MEX files.
%
% By default both mex files of the project are built
%   highsmex      - the solver interface, used by callhighs
%   highsmex_iis  - the IIS interface, used by callhighs_iis
% Pass mexSrcFiles to build only some of them, e.g.
%   make_highsmex('1.15.1', 'highsmex.cpp')
%
% Author: Savyasachi Singh
% Modified by: Ray Zimmerman
%
% Covered by the MIT License (see LICENSE file for details).
% See https://github.com/savyasachi/HiGHSMEX for more information.

%% Inputs
if nargin < 1 || isempty(highs_ver)
    highs_ver = '1.15.1';
end
if nargin < 2 || isempty(mexSrcFiles)
    mexSrcFiles = {'highsmex.cpp', 'highsmex_iis.cpp'};
elseif ~iscell(mexSrcFiles)
    mexSrcFiles = {mexSrcFiles};
end

% Path to HiGHS official binary HiGHS distribution, including HiPO solver.
% Extract archive beside HiGHSMEX repo.
if ispc
    arch = 'x86_64-windows';        % Windows
elseif ismac
    arch =  'arm-apple';            % macOX
elseif isunix
    arch =  'x86_64-linux-gnu';     % Linux
end
highsInstallDir = fullfile('..', ...
    sprintf('highs-%s-%s-static-apache', highs_ver, arch));

%% Paths to HiGHS include & lib dirs
highsIncludeDir = fullfile(highsInstallDir, 'include', 'highs');
highsLibDir = fullfile(highsInstallDir, 'lib');

%% Platform specific compiler, linker flags
compilerInfo=mex.getCompilerConfigurations('C++', 'Selected');
compilerVendor=lower(compilerInfo.Manufacturer);
switch compilerVendor
    case 'microsoft'
        compflags = {
            '-lopenblas', '-v', ...
            'COMPFLAGS="$COMPFLAGS /std:c++20  /W3 "', ...
        };
    case 'apple'
        compflags = { '-lz', '-v',
            'CXXFLAGS="$CXXFLAGS -std=c++20 -mmacosx-version-min=13.4 "', ...
            'LDFLAGS="$LDFLAGS -mmacosx-version-min=13.4 -Wl,-framework,Accelerate"', ...
        };
    case 'gnu'
        compflags = {
            '-lopenblas', '-lz', '-v', ...
            'COMPFLAGS="$COMPFLAGS -std=c++20 -Wall"', ...
            'LDFLAGS="$LDFLAGS -static-libstdc++ -static-libgcc"', ...
        };
end
if vstr2num(highs_ver) > 1.015
    compflags = {'-lhighs_extras', compflags{:}};
end

%% Build mex files
for k = 1:length(mexSrcFiles)
    mexSrcFilePath = fullfile('.', mexSrcFiles{k});
    fprintf('\n=== Building %s ===\n', mexSrcFiles{k});
    mex(mexSrcFilePath, '-R2018a', sprintf('-I"%s"', highsIncludeDir), ...
        sprintf('-L"%s"', highsLibDir), '-lhighs', compflags{:});
end


function num = vstr2num(vstr)
% Converts version string to numerical value suitable for < or > comparisons
% E.g. '1.15.1' -->  1.015001
pat = '\.?(\d+)';
[s,e,tE,m,t] = regexp(vstr, pat);
b = 1;
num = 0;
for k = 1:length(t)
    num = num + b * str2num(t{k}{1});
    b = b / 1000;
end
