function nFail = test_callhighs(nRandom)
% Test suite for callhighs.
%
% The suite checks the interface of callhighs, and then calls validatehighsmex
% to verify the solutions themselves: every solution is checked against the
% model it came from, and where Gurobi is available the model status and the
% optimal objective value are compared against it.
%
% USAGE:
% nFail = test_callhighs()
% nFail = test_callhighs(nRandom)
%
% INPUTS:
% nRandom - Number of randomly generated models passed on to validatehighsmex.
%           Pass 0 to run the fixed models only. The default is 10.
%
% OUTPUTS:
% nFail - Number of failed checks. Zero means everything passed.
%
% Gurobi is optional, see findgurobi. Without it the solver comparison is
% skipped, which is not a failure.
%
% EXAMPLES:
% nFail = test_callhighs();
% assert(nFail == 0)
%
% See also callhighs, validatehighsmex, test_callhighs_iis.
%
% Covered by the MIT License (see LICENSE file for details).
% See https://github.com/savyasachi/HiGHSMEX for more information.

arguments
    nRandom (1,1) double {mustBeNonnegative, mustBeInteger} = 10
end

nFail = 0;
oldWarn = warning("off", "highs:mex");
restoreWarn = onCleanup(@() warning(oldWarn));

fprintf("HiGHS %s, HiGHSMEX %s, HighsInt %s\n\n", ...
    callhighs("ver"), verHiGHSMEX(), callhighs("intType"));

%% ---------------------------------------------------------------- interface
fprintf("== interface ==\n");
intType = callhighs("intType");
nFail = nFail + check("intType is an integer class", ismember(intType, {'int32', 'int64'}));
defopts = callhighs("defopts");
nFail = nFail + check("defopts is a struct with many options", isstruct(defopts) && numel(fieldnames(defopts)) > 50);
nFail = nFail + check("defopts exposes presolve", isfield(defopts, "presolve"));

quiet = highsoptset("output_flag", 0);
c = [1 1]; A = [0 1; 1 2; 3 2];
L = [-inf 5 6]; U = [7 15 inf]; l = [0 1]; u = [4 inf];

% output arity
[soln, info, opts, basis] = callhighs(c, A, L, U, l, u, [], [], quiet);
nFail = nFail + check("four outputs are structs", ...
    isstruct(soln) && isstruct(info) && isstruct(opts) && isstruct(basis));
nFail = nFail + check("solution has the HighsSolution fields", ...
    all(isfield(soln, {'value_valid', 'dual_valid', 'col_value', 'col_dual', 'row_value', 'row_dual'})));
nFail = nFail + check("basis has the HighsBasis fields", ...
    all(isfield(basis, {'valid', 'col_status', 'row_status'})));
nFail = nFail + check("col_value has one entry per column", numel(soln.col_value) == size(A, 2));
nFail = nFail + check("row_value has one entry per row", numel(soln.row_value) == size(A, 1));
nFail = nFail + check("model status is Optimal", strcmp(info.model_status_string, "Optimal"));
nFail = nFail + check("opts reports the option that was set", opts.output_flag == false);

% a sparse A and the {i,j,v,nr,nc} cell form must agree with the dense form
cellA = cell(1, 5);
[cellA{1}, cellA{2}, cellA{3}] = find(sparse(A));
[cellA{4}, cellA{5}] = size(A);
s2 = callhighs(c, sparse(A), L, U, l, u, [], [], quiet);
s3 = callhighs(c, cellA, L, U, l, u, [], [], quiet);
nFail = nFail + check("sparse A gives the same solution as dense A", isequaln(soln.col_value, s2.col_value));
nFail = nFail + check("cell A gives the same solution as dense A", isequaln(soln.col_value, s3.col_value));

% objSense
sMin = callhighs(c, A, L, U, l, [4 10], [], [], quiet, "min");
sMax = callhighs(c, A, L, U, l, [4 10], [], [], quiet, "max");
nFail = nFail + check("max gives an objective at least as large as min", ...
    c * sMax.col_value >= c * sMin.col_value - 1e-9);

% integrality
sInt = callhighs(c, A, L, U, l, [4 10], [], ["i"; "i"], quiet);
nFail = nFail + check("integrality yields integral values", ...
    max(abs(sInt.col_value - round(sInt.col_value))) < 1e-6);

% hot start with a solution struct, a solution vector and a basis
Q = [1 -1; -1 2];
[sQ, ~, ~, bQ] = callhighs([0 0], [1 1], [], 3, [0 0], [10 10], Q, [], quiet);
sHotStruct = callhighs([0 0], [1 1], [], 3, [0 0], [10 10], Q, [], quiet, [], sQ, bQ);
sHotVector = callhighs([0 0], [1 1], [], 3, [0 0], [10 10], Q, [], quiet, [], sQ.col_value, bQ);
nFail = nFail + check("hot start from a solution struct reproduces the solution", ...
    norm(sHotStruct.col_value - sQ.col_value) < 1e-6);
nFail = nFail + check("hot start from a solution vector reproduces the solution", ...
    norm(sHotVector.col_value - sQ.col_value) < 1e-6);

% multi-objective
clear mo
mo(1) = struct('weight', -1, 'offset', -1, 'coefficients', [1, 1], ...
    'abs_tolerance', 0, 'rel_tolerance', 0, 'priority', cast(0, intType));
mo(2) = struct('weight', 1e-4, 'offset', 0, 'coefficients', [1, 0], ...
    'abs_tolerance', 0, 'rel_tolerance', 0, 'priority', cast(0, intType));
Amo = [3 1; 1 1; 1 2];
sMo = callhighs(mo, Amo, -inf(3, 1), [18; 8; 14], [0; 0], inf(2, 1), [], ["c"; "c"], ...
    highsoptset("blend_multi_objectives", 1, "output_flag", 0));
nFail = nFail + check("multi-objective LP with blending solves to [2; 6]", ...
    norm(sMo.col_value - [2; 6]) < 1e-6);

% options of each of the four HiGHS types round-trip
o = struct();
o.output_flag = false;
o.simplex_iteration_limit = cast(1234, intType);
o.primal_feasibility_tolerance = 1e-8;
o.solver = "simplex";
[~, ~, optsAll] = callhighs(c, A, L, U, l, u, [], [], o);
nFail = nFail + check("bool option round-trips", optsAll.output_flag == false);
nFail = nFail + check("integer option round-trips", optsAll.simplex_iteration_limit == 1234);
nFail = nFail + check("double option round-trips", optsAll.primal_feasibility_tolerance == 1e-8);
nFail = nFail + check("string option round-trips", strcmp(optsAll.solver, "simplex"));

% errors
nFail = nFail + check("rejects an unknown option", ...
    errorContains(@() callhighs(c, A, L, U, l, u, [], [], struct('no_such_option', 1)), ...
    "not a legal HiGHS option"));
nFail = nFail + check("rejects an invalid c", ...
    errorContains(@() callhighs({1, 2, 3}, A, L, U, l, u), "first input argument"));
nFail = nFail + check("rejects an invalid command string", ...
    errorContains(@() callhighs("nonsense"), "Input string is invalid"));
nFail = nFail + check("rejects a malformed basis struct", ...
    errorContains(@() callhighs(c, A, L, U, l, u, [], [], quiet, [], [], struct('valid', true)), "basis struct"));
nFail = nFail + check("rejects a wrong length for L", ...
    errorContains(@() callhighs(c, A, [1 2], U, l, u, [], [], quiet), "third input argument"));

%% ------------------------------------------------------- solution validation
fprintf("\n== solution validation ==\n");
nFail = nFail + validatehighsmex(nRandom, true);

fprintf("\n%s: %d failed check(s)\n", string(mfilename), nFail);

end


%% ================================================================ functions

function n = check(name, tf)
if tf
    fprintf("  PASS  %s\n", name);
    n = 0;
else
    fprintf("  FAIL  %s\n", name);
    n = 1;
end
end

% ----------------------------------------------------------------------- %

function tf = errorContains(fcn, needle)
% One output is requested, because callhighs rejects a call with no output
% argument before it validates any of its inputs.
try
    [~] = fcn();
    tf = false;
catch ME
    tf = contains(ME.message, needle, "IgnoreCase", true);
end
end

% EOF
