function varargout = callhighs_iis(varargin)
% MATLAB interface to the IIS facility of the HiGHS optimization library.
% https://github.com/ERGO-Code/HiGHS
%
% An irreducible infeasible subsystem (IIS) is a subset of the rows and
% columns of an infeasible model, together with a subset of their bounds,
% that is itself infeasible, and that becomes feasible as soon as any one of
% those bounds is removed. It answers the question "why is my model
% infeasible?".
%
% For the program with variable x
%
% subject to
%  L <= A ⋅ x <= U
%  l <= x <= u
% where,
% A is a m x n matrix, L, U are m x 1 vectors, and l, u are n x 1 vectors,
% callhighs_iis returns the rows and columns of that system that make it
% infeasible. Note that the objective plays no role in infeasibility, hence
% neither the cost vector c nor the Hessian Q is used in the calculation.
%
% CAUTION: The HiGHS IIS facility is available for LINEAR programs only. It
% cannot be used for mixed integer or quadratic programs, hence
% callhighs_iis accepts neither an integrality vector nor a Hessian.
% See https://ergo-code.github.io/HiGHS/stable/guide/advanced/
%
% USAGE:
% 1) ver = callhighs_iis("ver")
%    Returns the HiGHS version string.
%
% 2) intType = callhighs_iis("intType")
%    Returns the MATLAB class corresponding to HighsInt type. This will be
%    "int32" or "int64".
%
% 3) defopts = callhighs_iis("defopts")
%    Returns the default values for all the user settable options of HiGHS
%    as a MATLAB struct. All the HiGHS options are listed at the following
%    webpage.
%    https://ergo-code.github.io/HiGHS/stable/options/definitions/
%
% 4) [iis, submodel, info, opts] = callhighs_iis(c, A, L, U, l, u, options)
%    Determines an IIS. First two inputs (c and A) are required and the rest
%    are optional. Pass [] for any input argument to use the default value
%    for that argument.
%
% INPUTS:
% The first six input arguments are identical to those of callhighs, so that
% an infeasible model can be handed over from callhighs to callhighs_iis
% unchanged.
% c - Vector of column costs. It is not used in the IIS calculation and is
%     accepted only for compatibility with callhighs. Pass [] to omit it, in
%     which case the number of columns is taken from A.
% A - Linear inequality constraint matrix. It can be full or sparse.
%                             - OR -
%     It should be a cell-array of 5 elements in the following format
%     {i, j, v, nr, nc} where, [i, j, v]=find(A) and [nr, nc]=size(A).
%     Pass [] to omit the linear inequality constraints, in which case c
%     must not be empty.
% L - Lower bound vector for the linear inequality constraint. Pass [] to
%     set the lower bound to negative infinity.
% U - Upper bound vector for the linear inequality constraint. Pass [] to
%     set the upper bound to infinity.
% l - Lower bound vector for the optimization variable. Pass [] to set the
%     lower bound to negative infinity.
% u - Upper bound vector for the optimization variable. Pass [] to set the
%     upper bound to infinity.
% options - MATLAB struct with values for the HiGHS user-settable options.
%           Pass [] to use the default values for the user-settable options.
%           See https://ergo-code.github.io/HiGHS/stable/options/definitions/
%           The options relevant to the IIS calculation are
%           iis_strategy   - Bit map controlling how the IIS is computed
%                            0 - light strategy, i.e. the cheap trivial
%                                checks only (inconsistent bounds, empty
%                                infeasible rows, row value bounds)
%                            2 - form a mutually infeasible set of rows by
%                                solving elasticity LPs
%                            4 - reduce that set to an irreducible one
%                            8 - prioritize fewer columns over fewer rows
%                                while reducing
%                            Unlike HiGHS, whose default is 0,
%                            callhighs_iis defaults to 2+4=6, i.e. a full
%                            IIS calculation. Pass iis_strategy explicitly
%                            to override this. Note that it must be of the
%                            class returned by callhighs_iis("intType").
%           iis_time_limit - Time limit in seconds for the IIS calculation.
%
% OUTPUTS:
% iis - Struct describing the IIS. Its fields mirror the data members of the
%       HighsIis class of HiGHS. See
%       https://ergo-code.github.io/HiGHS/stable/structures/classes/HighsIis/
%       valid      - True if the contents of the struct are known to be
%                    correct. Always check this before using the other
%                    fields.
%       status     - Numeric status of the IIS calculation.
%       status_string - The same status as a string
%                    "feasible"    - The model is feasible, hence no IIS
%                                    exists and the index fields are empty.
%                    "unknown"     - No infeasible subsystem was determined.
%                    "timeLimit"   - iis_time_limit was reached.
%                    "reducible"   - An infeasible subsystem was found, but
%                                    it was not reduced to an irreducible
%                                    one (e.g. iis_strategy without bit 4).
%                    "irreducible" - A true IIS was found.
%       strategy   - The value of the iis_strategy option that was used.
%       row_index  - Indices (into the rows of A) of the constraints in the
%                    IIS. One based, so A(iis.row_index, :) selects them.
%       col_index  - Indices (into the columns of A) of the variables in the
%                    IIS. One based.
%       row_bound  - Vector of strings, one per entry of row_index, naming
%                    the bound of that row that takes part in the IIS
%                    "lower"   - Only the lower bound L is involved
%                    "upper"   - Only the upper bound U is involved
%                    "boxed"   - Both bounds are involved
%                    "free"    - Neither bound is involved
%                    "null"    - No bound status has been determined
%                    "dropped" - The row was dropped during the reduction
%       col_bound  - As row_bound, but for the bounds l and u of the
%                    variables in col_index.
%       row_status - Vector of strings with one entry per row of A
%                    "inConflict"      - The row is part of the IIS
%                    "maybeInConflict" - The row is part of an infeasible
%                                        subsystem that was not reduced to
%                                        an irreducible one
%                    "notInConflict"   - The row is not part of the IIS
%       col_status - As row_status, but with one entry per column of A.
%       info       - Struct with the number of LPs solved and the simplex
%                    iteration counts and times of the IIS calculation.
% submodel - The IIS itself, as a standalone infeasible model, laid out like
%       the input arguments of callhighs so it can be passed straight back
%       to callhighs or callhighs_iis. Its fields are
%       c        - Vector of column costs. These are zero, since costs play
%                  no role in infeasibility.
%       A        - Sparse constraint matrix, equal to
%                  A(iis.row_index, iis.col_index) of the original model.
%       L, U     - Row bounds. Bounds that are not part of the IIS (see
%                  iis.row_bound) are infinite here.
%       l, u     - Column bounds. Bounds that are not part of the IIS (see
%                  iis.col_bound) are infinite here.
%       rowIndex - Same as iis.row_index. Maps the rows of the submodel back
%                  to the rows of the original model.
%       colIndex - Same as iis.col_index. Maps the columns of the submodel
%                  back to the columns of the original model.
% info - Solution information struct returned by HiGHS for the last LP that
%        was solved during the IIS calculation. See
%        https://ergo-code.github.io/HiGHS/stable/structures/structs/HighsInfo/
% opts - Struct containing the values of user settable options used by
%        HiGHS. See https://ergo-code.github.io/HiGHS/stable/options/definitions/
%
% NOTE:
% The IIS calculation requires the solution of multiple LPs and is therefore
% considerably more expensive than a single solve. Use the iis_time_limit
% option to bound the effort.
%
% A model whose infeasibility HiGHS can see while the model is being passed
% to it, such as a variable with contradictory bounds, produces a warning
% before the IIS is returned. That is expected, and the IIS is still valid.
%
% The warning ID for all the warnings issued by callhighs_iis is highs:mex.
% To turn off the warnings use warning("off", "highs:mex").
%
% CAUTION:
% Do NOT call the highsmex_iis function directly in MATLAB, otherwise it
% would lead to a crash. Always use callhighs_iis function.
%
% EXAMPLES:
% See MATLAB script example_callhighs_iis.m for examples.
%
% See also callhighs, highsoptset.
%
% Covered by the MIT License (see LICENSE file for details).
% See https://github.com/savyasachi/HiGHSMEX for more information.


% Out of process execution of highsmex_iis function.
% https://www.mathworks.com/help/matlab/matlab_external/out-of-process-execution-of-c-mex-functions.html

persistent mh

useMexHostByDefault = true;
% useMexHostByDefault = false;

if nargin > 1 && issparse(varargin{2})
    varargin{2} = sparseMatrixToCell(varargin{2});
end

% The mex function always returns at least the iis struct, so that calling
% callhighs_iis without an output argument displays it via ans.
nout = max(nargout, 1);

if useMexHostByDefault && ~isdeployed
    %% OUT-OF-PROCESS (desktop MATLAB only)
    if ~(isa(mh, "matlab.mex.MexHost") && isvalid(mh))
        mh = mexhost;
    end
    [varargout{1:nout}] = feval(mh, "highsmex_iis", varargin{:});
else
    %% IN-PROCESS (for compiled MATLAB code)
    [varargout{1:nout}] = highsmex_iis(varargin{:});
end

% The mex function returns the constraint matrix of the submodel in the
% {i, j, v, nr, nc} format. Assemble it into a MATLAB sparse matrix.
if nout > 1 && isstruct(varargout{2}) && iscell(varargout{2}.A)
    varargout{2}.A = cellToSparseMatrix(varargout{2}.A);
end

% ----------------------------------------------------------------------- %

function c = sparseMatrixToCell(A)

c = cell(1, 5);
[c{1}, c{2}, c{3}] = find(A);
[c{4}, c{5}] = size(A);

% ----------------------------------------------------------------------- %

function A = cellToSparseMatrix(c)

A = sparse(c{1}, c{2}, c{3}, c{4}, c{5});

% EOF
