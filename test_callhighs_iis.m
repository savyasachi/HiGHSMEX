function nFail = test_callhighs_iis(nRandom)
% Test suite for callhighs_iis.
%
% The suite checks the interface of callhighs_iis, and verifies every IIS it
% produces against the definition of an IIS using validateiis. If Gurobi is
% available it additionally cross-checks against gurobi_iis.
%
% Because an IIS is not unique, the cross-check does not require the two
% solvers to return the same subsystem. Instead, each result is verified
% independently with BOTH solvers, so that neither is its own referee.
% Differences are reported for information only and are not failures.
%
% USAGE:
% nFail = test_callhighs_iis()
% nFail = test_callhighs_iis(nRandom)
%
% INPUTS:
% nRandom - Number of randomly generated infeasible models to test. Pass 0
%           to run the fixed models only. The default is 10.
%
% OUTPUTS:
% nFail - Number of failed checks. Zero means everything passed.
%
% GUROBI:
% The Gurobi cross-check is run when the Gurobi MATLAB interface can be
% located, see findgurobi for how that is done via the GUROBI_HOME environment
% variable. Without a working and licensed installation the cross-check is
% skipped, which is not a failure.
%
% EXAMPLES:
% nFail = test_callhighs_iis();
% assert(nFail == 0)
%
% See also callhighs_iis, validateiis, findgurobi, test_callhighs.
%
% Covered by the MIT License (see LICENSE file for details).
% See https://github.com/savyasachi/HiGHSMEX for more information.

arguments
    nRandom (1,1) double {mustBeNonnegative, mustBeInteger} = 10
end

nFail = 0;
o = struct("output_flag", false);

% HiGHS legitimately warns when it spots the infeasibility while the model is
% still being passed to it. That is expected here and would only be noise.
oldWarn = warning("off", "highs:mex");
restoreWarn = onCleanup(@() warning(oldWarn));

fprintf("HiGHS %s, HiGHSMEX %s, HighsInt %s\n\n", ...
    callhighs_iis("ver"), verHiGHSMEX(), callhighs_iis("intType"));

%% ---------------------------------------------------------------- interface
fprintf("== interface ==\n");
intType = callhighs_iis("intType");
nFail = nFail + check("ver agrees with callhighs", strcmp(callhighs_iis("ver"), callhighs("ver")));
defopts = callhighs_iis("defopts");
nFail = nFail + check("defopts exposes iis_strategy", isfield(defopts, "iis_strategy"));

c = [1 1];
A = [1 1; 1 1; 1 -1];
L = [4 -inf -inf]; U = [inf 2 1]; l = [0 0]; u = [10 10];
[iis, sub, info, opts] = callhighs_iis(c, A, L, U, l, u, o);

nFail = nFail + check("four outputs are structs", ...
    isstruct(iis) && isstruct(sub) && isstruct(info) && isstruct(opts));
nFail = nFail + check("default iis_strategy is 6", double(iis.strategy) == 6);
nFail = nFail + check("status is irreducible", strcmp(iis.status_string, "irreducible"));
nFail = nFail + check("row_status covers every row", numel(iis.row_status) == size(A, 1));
nFail = nFail + check("col_status covers every column", numel(iis.col_status) == size(A, 2));
nFail = nFail + check("row_bound matches row_index", numel(iis.row_bound) == numel(iis.row_index));
nFail = nFail + check("col_bound matches col_index", numel(iis.col_bound) == numel(iis.col_index));
nFail = nFail + check("rows in the IIS are flagged inConflict", ...
    all(iis.row_status(iis.row_index) == "inConflict"));
nFail = nFail + check("indices are one based", ...
    all(iis.row_index >= 1) && all(iis.row_index <= size(A, 1)));

% Submodel
nFail = nFail + check("submodel A is sparse", issparse(sub.A));
nFail = nFail + check("submodel A is the restriction of A", ...
    isequal(full(sub.A), A(iis.row_index, iis.col_index)));
nFail = nFail + check("submodel costs are zero", all(sub.c == 0));
nFail = nFail + check("submodel index maps agree with the iis struct", ...
    isequal(sub.rowIndex, iis.row_index) && isequal(sub.colIndex, iis.col_index));
[~, subInfo] = callhighs(sub.c, sub.A, sub.L, sub.U, sub.l, sub.u, [], [], o);
nFail = nFail + check("submodel is infeasible", strcmp(subInfo.model_status_string, "Infeasible"));

% iis_strategy is honoured
for s = [0 2 6]
    oS = o; oS.iis_strategy = cast(s, intType);
    iisS = callhighs_iis(c, A, L, U, l, u, oS);
    nFail = nFail + check("iis_strategy " + s + " is honoured", double(iisS.strategy) == s);
end

% Errors
nFail = nFail + check("rejects empty c together with empty A", ...
    errorContains(@() callhighs_iis([], [], [], [], [], []), "First input argument"));
nFail = nFail + check("rejects a column count mismatch", ...
    errorContains(@() callhighs_iis([1 1], [1 1 1], [], [], [], []), "columns"));
nFail = nFail + check("rejects an unknown option", ...
    errorContains(@() callhighs_iis([1 1], [1 1], [], [], [], [], struct("no_such_option", 1)), ...
    "not a legal HiGHS option"));

%% ------------------------------------------------------------------- models
M = testmodels();
for k = 1:nRandom
    m = randommodel(k);
    if ~isempty(m), M(end+1) = m; end %#ok<AGROW>
end

fprintf("\n== IIS validity (%d models) ==\n", numel(M));
for k = 1:numel(M)
    m = M(k);
    iis = callhighs_iis(m.c, m.A, m.L, m.U, m.l, m.u, o);
    [tf, msg] = validateiis(iis, m.A, m.L, m.U, m.l, m.u);
    nFail = nFail + check(m.name + " -> " + describe(iis), tf, msg);
end

%% ----------------------------------------------------------- gurobi compare
fprintf("\n== cross-check against gurobi_iis ==\n");
[haveGurobi, gurobiWhere] = findgurobi();
if ~haveGurobi
    fprintf("  SKIP  %s\n", gurobiWhere);
else
    fprintf("  using %s\n", gurobiWhere);
    nSame = 0;
    for k = 1:numel(M)
        m = M(k);
        iis = callhighs_iis(m.c, m.A, m.L, m.U, m.l, m.u, o);
        [gL, gU, gl, gu, gRows, gCols] = gurobiiis(m.A, m.L, m.U, m.l, m.u);

        % Every result must survive the definition check under BOTH solvers
        byGurobi = @isInfeasibleGurobi;
        [t1, m1] = validateiis(iis, m.A, m.L, m.U, m.l, m.u);
        [t2, m2] = validateiis(iis, m.A, m.L, m.U, m.l, m.u, byGurobi);
        [t3, m3] = validateiis(m.A, gL, gU, gl, gu);
        [t4, m4] = validateiis(m.A, gL, gU, gl, gu, byGurobi);
        nFail = nFail + check(m.name + ": HiGHS IIS valid per HiGHS", t1, m1);
        nFail = nFail + check(m.name + ": HiGHS IIS valid per Gurobi", t2, m2);
        nFail = nFail + check(m.name + ": Gurobi IIS valid per HiGHS", t3, m3);
        nFail = nFail + check(m.name + ": Gurobi IIS valid per Gurobi", t4, m4);

        % Agreement is informational: an IIS is not unique
        hRows = sort(iis.row_index(:))';
        hCols = sort(iis.col_index(iis.col_bound ~= "free"))';
        if isequal(hRows, gRows) && isequal(hCols, gCols)
            nSame = nSame + 1;
        else
            fprintf("  note  %s: HiGHS rows=%s cols=%s vs Gurobi rows=%s cols=%s (both valid, IIS is not unique)\n", ...
                m.name, brief(hRows), brief(hCols), brief(gRows), brief(gCols));
        end
    end
    fprintf("  identical IIS in %d of %d models\n", nSame, numel(M));
end

fprintf("\n%s: %d failed check(s)\n", string(mfilename), nFail);


%% ================================================================ functions

function M = testmodels()
% Infeasible models with a known conflict, covering the interesting shapes.

M = model("bound conflict, no rows", ...
    [1 1], [], [], [], [0 2], [4 1]);
M(end+1) = model("two conflicting rows", ...
    [1 1], [1 1; 1 1; 1 -1], [4 -inf -inf], [inf 2 1], [0 0], [10 10]);
M(end+1) = model("row bound vs column bound", ...
    [0 0], [1 0; 0 1], [3 -inf], [inf 0], [0 0], [1 5]);
M(end+1) = model("inconsistent range row", ...
    [0 0], [1 1; 1 -1], [4 -inf], [2 1], [0 0], [10 10]);
M(end+1) = model("two independent conflicts", ...
    [], [1 1 0 0; 1 1 0 0; 0 0 1 1; 0 0 1 1], [4 -inf 4 -inf], [inf 2 inf 2], ...
    [0 0 0 0], [10 10 10 10]);

% Sparse chain, rows are x_k + x_(k+1). Row 50 demands a large sum while row
% 51 together with the fixed x_52 forces x_51 to be small.
n = 200;
As = spdiags(ones(n, 2), [0 1], n-1, n);
Ls = -inf(n-1, 1); Us = inf(n-1, 1); ls = zeros(n, 1); us = ones(n, 1);
Ls(50) = 1.9; Us(51) = 0.1; us(52) = 0;
M(end+1) = model("sparse chain 199x200", [], As, Ls, Us, ls, us);

% ----------------------------------------------------------------------- %

function m = randommodel(seed)
% A random model whose infeasibility needs a genuine LP argument rather than
% bound propagation on a single row. Three rows satisfy a_k = a_i + a_j, with
% a_i*x >= b_i and a_j*x >= b_j but a_k*x <= b_i + b_j - 1. Every other row is
% given bounds it cannot violate over the variable box, so it can never be
% part of an IIS.

rng(100 + seed);
nc = randi([5 14]);
nr = randi([6 14]);
A = full(round(sprand(nr, nc, 0.4) * 6 - 3));
A(all(A == 0, 2), 1) = 1;                  % no empty rows
l = zeros(nc, 1);
u = ones(nc, 1) * randi([3 8]);

rowMin = @(a) sum(min(a(:) .* l, a(:) .* u));
rowMax = @(a) sum(max(a(:) .* l, a(:) .* u));

p = randperm(nr, 3);
i = p(1); j = p(2); k = p(3);
A(k, :) = A(i, :) + A(j, :);

L = -inf(nr, 1); U = inf(nr, 1);
for t = 1:nr
    L(t) = rowMin(A(t, :)) - 1;
    U(t) = rowMax(A(t, :)) + 1;
end
bi = rowMax(A(i, :)) - 0.5;
bj = rowMax(A(j, :)) - 0.5;
L(i) = bi;              U(i) = inf;
L(j) = bj;              U(j) = inf;
L(k) = -inf;            U(k) = bi + bj - 1;

% Each of the three rows must be satisfiable on its own, otherwise the IIS
% collapses to a single trivially violated row and tests nothing.
if bi < rowMin(A(i, :)) || bj < rowMin(A(j, :)) || bi + bj - 1 < rowMin(A(k, :)) + 0.5
    m = [];
    return
end
m = model(sprintf("random %d (%dx%d)", seed, nr, nc), [], A, L, U, l, u);
if ~isInfeasibleHighs(A, L, U, l, u)
    m = [];   % came out feasible after all
end

% ----------------------------------------------------------------------- %

function m = model(name, c, A, L, U, l, u)

m = struct("name", string(name), "c", c, "A", A, "L", L, "U", U, "l", l, "u", u);

% ----------------------------------------------------------------------- %

function [Liis, Uiis, liis, uiis, rows, cols] = gurobiiis(A, L, U, l, u)
% Compute an IIS with Gurobi and express it in the same terms as the HiGHS
% one, i.e. as bound vectors of the full model plus the index sets.

nc = numel(l);
if isempty(A), nr = 0; else, nr = size(A, 1); end
L = fillbound(L, nr, -inf); U = fillbound(U, nr, inf);
l = fillbound(l, nc, -inf); u = fillbound(u, nc, inf);

[gmodel, map] = togurobi(A, L, U, l, u);
gi = gurobi_iis(gmodel, struct("OutputFlag", 0));

Liis = -inf(nr, 1); Uiis = inf(nr, 1);
inIis = find(logical(gi.Arows(:)))';
for t = inIis
    r = map(t);
    if gmodel.sense(t) == '>' || gmodel.sense(t) == '=', Liis(r) = L(r); end
    if gmodel.sense(t) == '<' || gmodel.sense(t) == '=', Uiis(r) = U(r); end
end
liis = -inf(nc, 1); uiis = inf(nc, 1);
liis(logical(gi.lb(:))) = l(logical(gi.lb(:)));
uiis(logical(gi.ub(:))) = u(logical(gi.ub(:)));

rows = unique(map(inIis))';
cols = sort(find(logical(gi.lb(:)) | logical(gi.ub(:))))';

% ----------------------------------------------------------------------- %

function [gmodel, map] = togurobi(A, L, U, l, u)
% Gurobi's MATLAB model struct has no range constraints, so a two sided row
% becomes two constraints. map records the original row of each of them.

nc = numel(l);
if isempty(A)
    nr = 0;
    A = sparse(0, nc);
else
    nr = size(A, 1);
    A = sparse(A);
end
rows = zeros(0, 1); sense = ''; rhs = zeros(0, 1); map = zeros(0, 1);
for i = 1:nr
    hasL = isfinite(L(i));
    hasU = isfinite(U(i));
    if hasL && hasU && L(i) == U(i)
        rows(end+1, 1) = i; sense(end+1) = '='; rhs(end+1, 1) = L(i); map(end+1, 1) = i; %#ok<AGROW>
    else
        if hasL
            rows(end+1, 1) = i; sense(end+1) = '>'; rhs(end+1, 1) = L(i); map(end+1, 1) = i; %#ok<AGROW>
        end
        if hasU
            rows(end+1, 1) = i; sense(end+1) = '<'; rhs(end+1, 1) = U(i); map(end+1, 1) = i; %#ok<AGROW>
        end
    end
end
gmodel.A = A(rows, :);
if isempty(rows), gmodel.A = sparse(0, nc); end
gmodel.sense = sense(:);
gmodel.rhs = rhs;
gmodel.obj = zeros(nc, 1);
gmodel.lb = l(:);
gmodel.ub = u(:);
gmodel.modelsense = 'min';

% ----------------------------------------------------------------------- %

function tf = isInfeasibleGurobi(A, L, U, l, u)
% DualReductions off so that infeasible is never merged into infeasible or
% unbounded. The objective is zero here, so unbounded cannot occur anyway.

gmodel = togurobi(A, L, U, l, u);
r = gurobi(gmodel, struct("OutputFlag", 0, "DualReductions", 0));
tf = strcmp(r.status, 'INFEASIBLE');

% ----------------------------------------------------------------------- %

function tf = isInfeasibleHighs(A, L, U, l, u)

[~, info] = callhighs(zeros(numel(l), 1), A, L, U, l, u, [], [], struct("output_flag", false));
tf = strcmp(info.model_status_string, "Infeasible");

% ----------------------------------------------------------------------- %

function n = check(name, tf, msg)

if nargin < 3, msg = ""; end
if tf
    fprintf("  PASS  %s\n", name);
    n = 0;
else
    if strlength(msg) > 0
        fprintf("  FAIL  %s (%s)\n", name, msg);
    else
        fprintf("  FAIL  %s\n", name);
    end
    n = 1;
end

% ----------------------------------------------------------------------- %

function tf = errorContains(fcn, needle)

try
    fcn();
    tf = false;
catch ME
    tf = contains(ME.message, needle);
end

% ----------------------------------------------------------------------- %

function s = describe(iis)

s = sprintf("%s, %d row(s), %d column(s)", iis.status_string, ...
    numel(iis.row_index), numel(iis.col_index));

% ----------------------------------------------------------------------- %

function s = brief(v)

if isempty(v)
    s = "[]";
elseif numel(v) > 10
    s = string(numel(v)) + " entries";
else
    s = string(mat2str(v));
end

% ----------------------------------------------------------------------- %

function v = fillbound(v, n, default)

if isempty(v)
    v = repmat(default, n, 1);
else
    v = v(:);
end

% EOF
