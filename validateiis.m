function [tf, msg] = validateiis(varargin)
% Check that an irreducible infeasible subsystem really is one.
%
% An IIS is, by definition, a subsystem that
%   1) is infeasible, and
%   2) becomes feasible as soon as any single one of its bounds is removed.
% validateiis verifies both properties by solving one linear program per
% bound. It is solver agnostic, so it can also be used to check an IIS that
% was produced by something other than HiGHS.
%
% Note that an IIS is NOT unique. Two correct implementations may return
% different subsystems for the same model, so comparing an IIS against a
% reference result is not a meaningful test, whereas this check is.
%
% USAGE:
% 1) [tf, msg] = validateiis(iis, A, L, U, l, u)
%    Checks the IIS described by the struct iis, as returned by
%    callhighs_iis, against the model it was computed from.
%
% 2) [tf, msg] = validateiis(A, Liis, Uiis, liis, uiis)
%    Checks a subsystem given directly by its bound vectors. The vectors
%    have the length of the full model, and carry -inf / inf wherever the
%    corresponding bound is NOT part of the subsystem. Use this form for an
%    IIS that did not come from callhighs_iis.
%
% 3) [tf, msg] = validateiis(..., isInfeasible)
%    As above, but with a function handle used to decide feasibility, so
%    that the check can be run with a different solver than the one that
%    produced the IIS. It is called as
%        tf = isInfeasible(A, L, U, l, u)
%    and must return true when the linear program is infeasible. The
%    default uses callhighs.
%
% INPUTS:
% iis  - IIS struct as returned by callhighs_iis.
% A    - Linear inequality constraint matrix of the full model, full or
%        sparse. Pass [] if the model has no linear constraints.
% L, U - Row bound vectors of the full model.
% l, u - Column bound vectors of the full model.
% Liis, Uiis, liis, uiis - Bound vectors of the subsystem, see syntax 2.
% isInfeasible - Function handle, see syntax 3.
%
% OUTPUTS:
% tf  - True if the subsystem is a valid IIS.
% msg - Empty when tf is true, otherwise a description of the first
%       property that was violated.
%
% NOTE:
% The cost is one LP solve per finite bound of the subsystem, plus one for
% the subsystem itself. That is cheap for an IIS, which is usually small,
% but do not call this on a full model.
%
% An IIS that is empty because the model is feasible is reported as valid,
% provided the model really is feasible. This case can only be detected in
% syntax 1, where the full model is known.
%
% EXAMPLES:
% % Verify the IIS of an infeasible model
% c = [1 1]; A = [1 1; 1 1; 1 -1];
% L = [4 -inf -inf]; U = [inf 2 1]; l = [0 0]; u = [10 10];
% iis = callhighs_iis(c, A, L, U, l, u);
% [tf, msg] = validateiis(iis, A, L, U, l, u)
%
% See also callhighs_iis, callhighs.
%
% Covered by the MIT License (see LICENSE file for details).
% See https://github.com/savyasachi/HiGHSMEX for more information.

narginchk(5, 7);

% Optional trailing function handle deciding feasibility
isInfeasible = @defaultIsInfeasible;
if isa(varargin{end}, 'function_handle')
    isInfeasible = varargin{end};
    varargin(end) = [];
end

if isstruct(varargin{1})
    %% Syntax 1: IIS struct plus the model it came from
    if numel(varargin) ~= 6
        error("highs:mex", "Expected the iis struct to be followed by A, L, U, l and u.");
    end
    [iis, A, L, U, l, u] = deal(varargin{1:6});
    nc = numel(l);
    if isempty(A), nr = 0; else, nr = size(A, 1); end
    L = expand(L, nr, -inf); U = expand(U, nr, inf);
    l = expand(l, nc, -inf); u = expand(u, nc, inf);

    if ~iis.valid
        tf = false;
        msg = "The iis struct is not valid, so there is nothing to check.";
        return
    end

    isEmptyIis = isempty(iis.row_index) && isempty(iis.col_index);
    if isEmptyIis
        % No IIS exists only if the model itself is feasible
        tf = ~isInfeasible(A, L, U, l, u);
        if tf
            msg = "";
        else
            msg = "The iis is empty, but the model is infeasible.";
        end
        return
    end

    [Liis, Uiis] = subsystemBounds(iis.row_index, iis.row_bound, L, U, nr, "row");
    [liis, uiis] = subsystemBounds(iis.col_index, iis.col_bound, l, u, nc, "column");
else
    %% Syntax 2: subsystem bound vectors
    if numel(varargin) ~= 5
        error("highs:mex", "Expected A to be followed by Liis, Uiis, liis and uiis.");
    end
    [A, Liis, Uiis, liis, uiis] = deal(varargin{1:5});
    nc = numel(liis);
    if isempty(A), nr = 0; else, nr = size(A, 1); end
    Liis = expand(Liis, nr, -inf); Uiis = expand(Uiis, nr, inf);
    liis = expand(liis, nc, -inf); uiis = expand(uiis, nc, inf);
end

%% Property 1: the subsystem is infeasible
if ~isInfeasible(A, Liis, Uiis, liis, uiis)
    tf = false;
    msg = "The subsystem is not infeasible.";
    return
end

%% Property 2: removing any single bound makes it feasible
for i = 1:nr
    if isfinite(Liis(i))
        relaxed = Liis; relaxed(i) = -inf;
        if isInfeasible(A, relaxed, Uiis, liis, uiis)
            tf = false;
            msg = "The subsystem is still infeasible without the lower bound of row " + i + ", so it is not irreducible.";
            return
        end
    end
    if isfinite(Uiis(i))
        relaxed = Uiis; relaxed(i) = inf;
        if isInfeasible(A, Liis, relaxed, liis, uiis)
            tf = false;
            msg = "The subsystem is still infeasible without the upper bound of row " + i + ", so it is not irreducible.";
            return
        end
    end
end
for j = 1:nc
    if isfinite(liis(j))
        relaxed = liis; relaxed(j) = -inf;
        if isInfeasible(A, Liis, Uiis, relaxed, uiis)
            tf = false;
            msg = "The subsystem is still infeasible without the lower bound of column " + j + ", so it is not irreducible.";
            return
        end
    end
    if isfinite(uiis(j))
        relaxed = uiis; relaxed(j) = inf;
        if isInfeasible(A, Liis, Uiis, liis, relaxed)
            tf = false;
            msg = "The subsystem is still infeasible without the upper bound of column " + j + ", so it is not irreducible.";
            return
        end
    end
end

tf = true;
msg = "";

% ----------------------------------------------------------------------- %

function [lo, up] = subsystemBounds(indx, bound, lo0, up0, n, what)
% Scatter the bounds that take part in the IIS into full length vectors.

if numel(indx) ~= numel(bound)
    error("highs:mex", "The %s_index and %s_bound fields of the iis struct must have the same length.", what, what);
end
lo = -inf(n, 1);
up = inf(n, 1);
for t = 1:numel(indx)
    k = indx(t);
    switch bound(t)
        case "lower"
            lo(k) = lo0(k);
        case "upper"
            up(k) = up0(k);
        case "boxed"
            lo(k) = lo0(k);
            up(k) = up0(k);
        case {"free", "null", "dropped"}
            % Neither bound of this row or column takes part in the IIS
        otherwise
            error("highs:mex", "Unknown %s bound status ""%s"" in the iis struct.", what, bound(t));
    end
end

% ----------------------------------------------------------------------- %

function tf = defaultIsInfeasible(A, L, U, l, u)
% Decide feasibility with HiGHS. The objective is irrelevant here.

if isempty(A)
    [~, info] = callhighs(zeros(numel(l), 1), [], [], [], l, u, [], [], struct("output_flag", false));
else
    [~, info] = callhighs(zeros(numel(l), 1), A, L, U, l, u, [], [], struct("output_flag", false));
end
tf = strcmp(info.model_status_string, "Infeasible");

% ----------------------------------------------------------------------- %

function v = expand(v, n, default)

if isempty(v)
    v = repmat(default, n, 1);
else
    v = v(:);
end

% EOF
