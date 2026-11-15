function [tf, where] = findgurobi()
% Locate the Gurobi MATLAB interface and add it to the MATLAB search path.
%
% This is a helper for the cross-checks against Gurobi in validatehighsmex and
% test_callhighs_iis. Neither of them requires Gurobi; when it cannot be found
% they skip the comparison rather than fail.
%
% USAGE:
% [tf, where] = findgurobi()
%
% OUTPUTS:
% tf    - True if a usable and licensed Gurobi MATLAB interface is available.
%         On success it has been added to the MATLAB path.
% where - The folder that was used when tf is true, otherwise a message saying
%         why Gurobi is not available.
%
% The interface is looked for in the "matlab" subfolder of the installation
% named by the GUROBI_HOME environment variable, e.g.
%   Windows  GUROBI_HOME=C:\gurobi1302\win64
%   Linux    GUROBI_HOME=/opt/gurobi1302/linux64
% and, failing that, on the MATLAB search path. A found installation is then
% checked by solving a one variable model, so that an expired or missing
% licence is reported here rather than surfacing later as a confusing error.
%
% See also validatehighsmex, test_callhighs_iis.
%
% Covered by the MIT License (see LICENSE file for details).
% See https://github.com/savyasachi/HiGHSMEX for more information.

where = "";
home = getenv("GUROBI_HOME");
if strlength(home) > 0
    candidate = fullfile(home, "matlab");
    if isfolder(candidate) && isfile(fullfile(candidate, "gurobi.m"))
        addpath(candidate);
        where = candidate;
    end
end
if strlength(where) == 0
    if exist("gurobi", "file")
        where = string(fileparts(which("gurobi")));
    elseif strlength(home) == 0
        tf = false;
        where = "GUROBI_HOME is not set and gurobi is not on the MATLAB path.";
        return
    else
        tf = false;
        where = "No Gurobi MATLAB interface in """ + fullfile(home, "matlab") + """ and none on the MATLAB path.";
        return
    end
end

try
    probe = struct("A", sparse(1, 1, 1), "obj", 0, "sense", '>', "rhs", 1, ...
        "lb", 0, "ub", 2, "modelsense", 'min');
    gurobi(probe, struct("OutputFlag", 0));
catch ME
    tf = false;
    where = "Gurobi found in """ + where + """ but not usable: " + string(ME.message);
    return
end
tf = true;

% EOF
