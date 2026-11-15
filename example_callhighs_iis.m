% This script provides examples for the different use cases of the
% callhighs_iis function, which determines an irreducible infeasible
% subsystem (IIS) of an infeasible linear program.
% Run this script section by section. Each section provides examples for
% different features of the HiGHS IIS facility.
%
% Covered by the MIT License (see LICENSE file for details).
% See https://github.com/savyasachi/HiGHSMEX for more information.

%% Get version of HiGHS, and the MATLAB class corresponding to the HighsInt (C++ integer type)

clc, clearvars

vHighs = callhighs_iis("ver");
vHighsMex = verHiGHSMEX();
intType = callhighs_iis("intType");
fprintf('HiGHS version is v%s.\nHiGHSMEX version is v%s.\nHighsInt type is %s.\n', vHighs, vHighsMex, intType)
fprintf('Press ENTER to continue.\n'), pause, clc

%% Inconsistent bounds on a variable

% There are no constraints at all here, the model is infeasible because the
% bounds on x_2 contradict each other. HiGHS finds this with the cheap
% trivial check, without solving a single LP.
%
% HiGHS warns about the inconsistent bounds while the model is passed to it
% ("Col 1 has inconsistent bounds [2, 1]"), which is exactly the conflict we
% are asking about. Use warning("off", "highs:mex") to silence it.

% Min    f  =  x_1 + x_2
% 0 <= x_1 <= 4; 2 <= x_2 <= 1

c = [1 1];
l = [0 2]; u = [4 1];

iis = callhighs_iis(c, [], [], [], l, u);

fprintf('IIS status is "%s".\n', iis.status_string)
fprintf('Columns in the IIS: %s\n', mat2str(iis.col_index'))
fprintf('Bounds of those columns that are in conflict: %s\n', strjoin(iis.col_bound, ', '))
fprintf('Rows in the IIS: %s\n', mat2str(iis.row_index'))
fprintf('Press ENTER to continue.\n'), pause, clc

%% Conflicting constraints

% Min    f  =  x_1 + x_2
% s.t.   r_1:  4 <=  x_1 +  x_2
%        r_2:        x_1 +  x_2 <= 2
%        r_3:        x_1 -  x_2 <= 1
% 0 <= x_1 <= 10; 0 <= x_2 <= 10
%
% Rows r_1 and r_2 cannot both hold, r_3 has nothing to do with it.

c = [1 1];
A = [1 1; 1 1; 1 -1];
L = [4 -inf -inf]; U = [inf 2 1];
l = [0 0]; u = [10 10];

% Confirm that the model really is infeasible
[soln, info] = callhighs(c, A, L, U, l, u);
fprintf('Model status reported by callhighs is "%s".\n\n', info.model_status_string)

% Determine the IIS
[iis, submodel] = callhighs_iis(c, A, L, U, l, u);

fprintf('IIS status is "%s".\n', iis.status_string)
fprintf('Rows in the IIS: %s\n', mat2str(iis.row_index'))
fprintf('Bounds of those rows that are in conflict: %s\n', strjoin(iis.row_bound, ', '))
fprintf('Columns in the IIS: %s\n', mat2str(iis.col_index'))
fprintf('Bounds of those columns that are in conflict: %s\n\n', strjoin(iis.col_bound, ', '))

% row_status has one entry per row of A, which makes it convenient for
% tagging the rows of the original model.
for k = 1:size(A, 1)
    fprintf('Row %d is %s.\n', k, iis.row_status(k))
end
fprintf('Press ENTER to continue.\n'), pause, clc

%% Work with the returned submodel

% The submodel is the IIS as a standalone model, laid out like the input
% arguments of callhighs.

fprintf('The IIS has %d rows and %d columns.\n', size(submodel.A, 1), size(submodel.A, 2))
fprintf('Its constraint matrix is\n'), disp(full(submodel.A))
fprintf('Its row bounds are\n'), disp([submodel.L submodel.U])
fprintf('Its column bounds are\n'), disp([submodel.l submodel.u])

% The submodel is infeasible by construction
[~, infoSub] = callhighs(submodel.c, submodel.A, submodel.L, submodel.U, submodel.l, submodel.u);
fprintf('\nModel status of the submodel is "%s".\n', infoSub.model_status_string)

% Removing any one of its bounds makes it feasible. Drop the lower bound of
% the first row of the submodel.
Lrelaxed = submodel.L;
Lrelaxed(1) = -inf;
[~, infoRelaxed] = callhighs(submodel.c, submodel.A, Lrelaxed, submodel.U, submodel.l, submodel.u);
fprintf('After dropping the lower bound of row %d, the model status is "%s".\n', ...
    submodel.rowIndex(1), infoRelaxed.model_status_string)
fprintf('Press ENTER to continue.\n'), pause, clc

%% Set the HiGHS options for the IIS calculation

% callhighs_iis runs a full IIS calculation by default, i.e.
% iis_strategy = 2 (build a mutually infeasible set of rows) + 4 (reduce it
% to an irreducible one). Pass iis_strategy explicitly to change this.
% Note that integer valued options must be of the class returned by
% callhighs_iis("intType").

opts = struct();
opts.iis_strategy = cast(2, intType);   % Do not reduce to an irreducible system
opts.iis_time_limit = 10;               % Seconds
opts.output_flag = false;               % Silence the HiGHS log

iisReducible = callhighs_iis(c, A, L, U, l, u, opts);
fprintf('With iis_strategy = 2 the status is "%s" and %d rows are reported.\n', ...
    iisReducible.status_string, numel(iisReducible.row_index))

opts.iis_strategy = cast(2 + 4, intType);
iisIrreducible = callhighs_iis(c, A, L, U, l, u, opts);
fprintf('With iis_strategy = 6 the status is "%s" and %d rows are reported.\n', ...
    iisIrreducible.status_string, numel(iisIrreducible.row_index))
fprintf('Press ENTER to continue.\n'), pause, clc

%% Sparse constraint matrix, and a model with no IIS

% callhighs_iis accepts a sparse A, just like callhighs. The cost vector is
% not used in the IIS calculation, so it may be omitted.

n = 200;
A = spdiags(ones(n, 2), [0 1], n - 1, n);  % Rows are x_k + x_{k+1}
L = -inf(n - 1, 1); U = inf(n - 1, 1);
l = zeros(n, 1); u = ones(n, 1);
% Two constraints on the same pair of variables that cannot both hold
L(50) = 1.9;    % x_50 + x_51 >= 1.9
U(50) = inf;
L(51) = -inf;
U(51) = 0.1;    % x_51 + x_52 <= 0.1
u(52) = 0;      % x_52 = 0, so x_51 <= 0.1 and hence x_50 + x_51 < 1.9 is forced

iis = callhighs_iis([], A, L, U, l, u);
fprintf('IIS status is "%s".\n', iis.status_string)
fprintf('Rows in the IIS: %s\n', mat2str(iis.row_index'))
fprintf('Columns in the IIS: %s\n\n', mat2str(iis.col_index'))

% A feasible model has no IIS. HiGHS reports this rather than failing.
iisFeasible = callhighs_iis([], A, -inf(n - 1, 1), inf(n - 1, 1), l, u);
fprintf('For a feasible model the status is "%s" and the IIS is empty (%d rows, %d columns).\n', ...
    iisFeasible.status_string, numel(iisFeasible.row_index), numel(iisFeasible.col_index))

% EOF
