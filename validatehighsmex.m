function nFail = validatehighsmex(nRandom, verbose)
% Validate callhighs by comparing it against an installed Gurobi.
%
% For every model the two solvers must agree on the model status and, when the
% model is solved to optimality, on the optimal objective value. The optimal
% point itself is NOT compared, because it need not be unique; instead the
% solution returned by HiGHS is checked directly against the model, i.e. its
% bounds, rows and integrality are verified and the objective is recomputed
% from it. That catches a wrong solution even where the two solvers happen to
% pick different optima of equal value.
%
% Both solvers are asked for a zero MIP gap, so that mixed integer objectives
% are comparable rather than merely within the default 1e-4 relative gap.
%
% Gurobi is optional. If it cannot be found the comparison is skipped and only
% the model-independent checks of the HiGHS solution are run, which is not a
% failure. See findgurobi for how it is located via GUROBI_HOME.
%
% USAGE:
% nFail = validatehighsmex()
% nFail = validatehighsmex(nRandom)
% nFail = validatehighsmex(nRandom, verbose)
%
% INPUTS:
% nRandom - Number of randomly generated models to test, in addition to the
%           fixed ones. Pass 0 to run the fixed models only. Default 10.
% verbose - Print one line per model. Default true.
%
% OUTPUTS:
% nFail - Number of failed checks. Zero means everything passed.
%
% EXAMPLES:
% nFail = validatehighsmex();
% assert(nFail == 0)
%
% See also callhighs, test_callhighs, findgurobi.
%
% Covered by the MIT License (see LICENSE file for details).
% See https://github.com/savyasachi/HiGHSMEX for more information.

arguments
    nRandom (1,1) double {mustBeNonnegative, mustBeInteger} = 10
    verbose (1,1) logical = true
end

nFail = 0;
tolFeas = 1e-6;     % absolute tolerance when checking bounds and rows
tolObj  = 1e-6;     % relative tolerance when comparing objective values

oldWarn = warning("off", "highs:mex");
restoreWarn = onCleanup(@() warning(oldWarn));

[haveGurobi, gurobiWhere] = findgurobi();
if verbose
    if haveGurobi
        fprintf("  comparing against Gurobi in %s\n", gurobiWhere);
    else
        fprintf("  SKIP  Gurobi comparison: %s\n", gurobiWhere);
    end
end

M = fixedmodels();
for k = 1:nRandom
    M(end+1) = randommodel(k); %#ok<AGROW>
end

for k = 1:numel(M)
    m = M(k);

    % ---- HiGHS
    opts = highsoptset("output_flag", 0, "mip_rel_gap", 0, "mip_abs_gap", 0);
    [soln, info] = callhighs(m.c, m.A, m.L, m.U, m.l, m.u, m.Q, m.integrality, opts, m.objSense);
    hStatus = canonicalHighsStatus(info.model_status_string);

    % ---- the HiGHS solution must satisfy the model it was given
    if hStatus == "optimal"
        [ok, why] = checkSolution(m, soln.col_value, info.objective_function_value, tolFeas, tolObj);
        nFail = nFail + report(verbose, m.name + ": HiGHS solution satisfies the model", ok, why);
    end

    % ---- Gurobi
    if ~haveGurobi
        if verbose
            fprintf("  ....  %-28s HiGHS %s\n", m.name, hStatus);
        end
        continue
    end
    gmodel = togurobi(m);
    r = gurobi(gmodel, struct("OutputFlag", 0, "DualReductions", 0, "MIPGap", 0, "MIPGapAbs", 0));
    gStatus = canonicalGurobiStatus(r.status);

    nFail = nFail + report(verbose, m.name + ": status agrees (" + hStatus + ")", ...
        hStatus == gStatus, "HiGHS says " + hStatus + ", Gurobi says " + gStatus + " (" + string(r.status) + ")");

    if hStatus == "optimal" && gStatus == "optimal"
        ho = info.objective_function_value;
        go = r.objval;
        ok = abs(ho - go) <= tolObj * max(1, max(abs(ho), abs(go)));
        nFail = nFail + report(verbose, sprintf("%s: objective agrees (%.10g)", m.name, ho), ok, ...
            sprintf("HiGHS %.12g vs Gurobi %.12g, difference %.3g", ho, go, ho - go));
    end
end

if verbose
    fprintf("\nvalidatehighsmex: %d failed check(s) over %d models\n", nFail, numel(M));
end

end


%% ================================================================ functions

function M = fixedmodels()
% Models with a known shape, covering the paths through callhighs.

M = model("LP", [1 1], [0 1; 1 2; 3 2], [-inf 5 6], [7 15 inf], [0 1], [4 inf]);
M(end+1) = model("LP max", [1 1], [0 1; 1 2; 3 2], [-inf 5 6], [7 15 inf], [0 1], [4 10], [], [], "max");
M(end+1) = model("LP range rows", [1 -2], [1 1; 1 -1], [2 -3], [6 3], [0 0], [5 5]);
M(end+1) = model("LP equality row", [1 1], [1 1], 4, 4, [0 0], [10 10]);
M(end+1) = model("LP infeasible", [1 1], [1 1; 1 1], [4 -inf], [inf 2], [0 0], [10 10]);
M(end+1) = model("LP unbounded", [-1 -1], [1 -1], -inf, 1, [0 0], [inf inf]);
M(end+1) = model("MILP", [1 1], [0 1; 1 2; 3 2], [-inf 5 6], [7 15 inf], [0 1], [4 10], [], ["i"; "i"]);
M(end+1) = model("MILP mixed", [-3 -2 -1], [1 1 1; 2 1 0], [-inf -inf], [10 8], [0 0 0], [5 5 5], [], ["i"; "c"; "i"]);
M(end+1) = model("QP", [-1 -2], [], [], [], [-10 -10], [10 10], [4 1; 1 2]);
M(end+1) = model("QP constrained", [0.5 -0.5 0.01], [1 0 1; 1 1 0], [1 2], [inf inf], [0 0 0], [5 5 5], ...
    [2 0 1; 0 3 0; 1 0 2]);
M(end+1) = model("QP diagonal", [-1 -1 -1], [1 1 1], -inf, 2, [0 0 0], [10 10 10], diag([2 4 6]));

end

% ----------------------------------------------------------------------- %

function m = randommodel(seed)
% A random model that is feasible and bounded by construction, so that both
% solvers must report Optimal and the objective values are comparable. The
% rows are built around a point inside the variable box, and every variable is
% boxed, which rules out an unbounded objective.

rng(2000 + seed);
nc = randi([3 10]);
nr = randi([2 8]);
A = full(round(sprand(nr, nc, 0.5) * 8 - 4));
A(all(A == 0, 2), 1) = 1;                       % no empty rows

l = zeros(nc, 1);
u = randi([1 6], nc, 1);
x0 = l + (u - l) .* rand(nc, 1);                % a point that must stay feasible
r0 = A * x0;

L = -inf(nr, 1);
U = inf(nr, 1);
for i = 1:nr
    switch randi(3)
        case 1, L(i) = r0(i) - randi([0 3]);
        case 2, U(i) = r0(i) + randi([0 3]);
        case 3, L(i) = r0(i) - randi([0 3]); U(i) = r0(i) + randi([0 3]);
    end
end

c = round(randn(nc, 1) * 4, 2);
kind = randi(3);
Q = [];
integrality = [];
switch kind
    case 1
        name = "random LP %d (%dx%d)";
    case 2
        name = "random MILP %d (%dx%d)";
        integrality = repmat("c", nc, 1);
        integrality(rand(nc, 1) < 0.6) = "i";
        u = ceil(u);                            % integral bounds keep the model tidy
    case 3
        name = "random QP %d (%dx%d)";
        B = randn(nc);
        Q = B' * B + nc * eye(nc);              % symmetric positive definite, so convex
        Q = round(Q, 4);
end
m = model(sprintf(name, seed, nr, nc), c, A, L, U, l, u, Q, integrality);

end

% ----------------------------------------------------------------------- %

function m = model(name, c, A, L, U, l, u, Q, integrality, objSense)

if nargin < 8,  Q = []; end
if nargin < 9,  integrality = []; end
if nargin < 10, objSense = "min"; end
m = struct("name", string(name), "c", c, "A", A, "L", L, "U", U, "l", l, "u", u, ...
    "Q", Q, "integrality", integrality, "objSense", string(objSense));

end

% ----------------------------------------------------------------------- %

function [ok, why] = checkSolution(m, x, objReported, tolFeas, tolObj)
% Verify the HiGHS solution against the model, independently of any solver.

x = x(:);
nc = numel(m.c);
[L, U, l, u] = expandBounds(m, nc);

if numel(x) ~= nc
    ok = false; why = sprintf("solution has %d entries, expected %d", numel(x), nc); return
end
if any(~isfinite(x))
    ok = false; why = "solution contains non-finite entries"; return
end

% variable bounds
[worst, j] = max(max(l - x, x - u));
if worst > tolFeas
    ok = false; why = sprintf("column %d violates its bounds by %.3g", j, worst); return
end

% rows
if ~isempty(m.A)
    Ax = m.A * x;
    [worst, i] = max(max(L - Ax, Ax - U));
    if worst > tolFeas
        ok = false; why = sprintf("row %d violates its bounds by %.3g", i, worst); return
    end
end

% integrality
if ~isempty(m.integrality)
    isInt = m.integrality(:) == "i";
    if any(isInt)
        worst = max(abs(x(isInt) - round(x(isInt))));
        if worst > 1e-5
            ok = false; why = sprintf("an integer variable is off by %.3g", worst); return
        end
    end
end

% the reported objective must match the one implied by the solution
objFromX = m.c(:)' * x;
if ~isempty(m.Q)
    objFromX = objFromX + 0.5 * (x' * m.Q * x);
end
if abs(objFromX - objReported) > tolObj * max(1, max(abs(objFromX), abs(objReported)))
    ok = false;
    why = sprintf("objective from the solution is %.12g but HiGHS reports %.12g", objFromX, objReported);
    return
end

ok = true; why = "";

end

% ----------------------------------------------------------------------- %

function gmodel = togurobi(m)
% Translate the model into Gurobi's MATLAB struct.
%
% Two conversions are needed. Gurobi has no range constraints, so a row with
% two finite bounds becomes two constraints. And Gurobi's objective is
% x'*Q*x + c'*x whereas HiGHS uses 0.5*x'*Q*x + c'*x, so Q is halved. The full
% symmetric matrix is passed, not just a triangle, because Gurobi reads every
% stored entry rather than mirroring one triangle.

nc = numel(m.c);
[L, U, l, u] = expandBounds(m, nc);
if isempty(m.A)
    nr = 0;
    A = sparse(0, nc);
else
    nr = size(m.A, 1);
    A = sparse(m.A);
end

rows = zeros(0, 1); sense = ''; rhs = zeros(0, 1);
for i = 1:nr
    hasL = isfinite(L(i));
    hasU = isfinite(U(i));
    if hasL && hasU && L(i) == U(i)
        rows(end+1, 1) = i; sense(end+1) = '='; rhs(end+1, 1) = L(i); %#ok<AGROW>
    else
        if hasL, rows(end+1, 1) = i; sense(end+1) = '>'; rhs(end+1, 1) = L(i); end %#ok<AGROW>
        if hasU, rows(end+1, 1) = i; sense(end+1) = '<'; rhs(end+1, 1) = U(i); end %#ok<AGROW>
    end
end

gmodel.A = A(rows, :);
if isempty(rows), gmodel.A = sparse(0, nc); end
gmodel.sense = sense(:);
gmodel.rhs = rhs;
gmodel.obj = m.c(:);
gmodel.lb = l;
gmodel.ub = u;
gmodel.modelsense = char(m.objSense);

if ~isempty(m.Q)
    gmodel.Q = sparse(0.5 * m.Q);
end
if ~isempty(m.integrality)
    vtype = repmat('C', nc, 1);
    known = ["c", "i"];
    if ~all(ismember(m.integrality(:), known))
        error("highs:mex", "togurobi only handles the continuous and integer variable types.");
    end
    vtype(m.integrality(:) == "i") = 'I';
    gmodel.vtype = vtype;
end

end

% ----------------------------------------------------------------------- %

function [L, U, l, u] = expandBounds(m, nc)
% Fill in the defaults that callhighs applies to omitted bound arguments.

if isempty(m.A), nr = 0; else, nr = size(m.A, 1); end
L = fill(m.L, nr, -inf);
U = fill(m.U, nr, inf);
l = fill(m.l, nc, -inf);
u = fill(m.u, nc, inf);

end

function v = fill(v, n, default)
if isempty(v), v = repmat(default, n, 1); else, v = v(:); end
end

% ----------------------------------------------------------------------- %

function s = canonicalHighsStatus(s)
switch string(s)
    case "Optimal",     s = "optimal";
    case "Infeasible",  s = "infeasible";
    case "Unbounded",   s = "unbounded";
    otherwise,          s = "other:" + string(s);
end
end

function s = canonicalGurobiStatus(s)
switch string(s)
    case "OPTIMAL",     s = "optimal";
    case "INFEASIBLE",  s = "infeasible";
    case "UNBOUNDED",   s = "unbounded";
    otherwise,          s = "other:" + string(s);
end
end

% ----------------------------------------------------------------------- %

function n = report(verbose, name, ok, why)
if ok
    if verbose, fprintf("  PASS  %s\n", name); end
    n = 0;
else
    fprintf("  FAIL  %s (%s)\n", name, why);
    n = 1;
end
end

% EOF
