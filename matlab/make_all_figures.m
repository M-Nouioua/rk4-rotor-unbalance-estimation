%MAKE_ALL_FIGURES Render every manuscript figure to analysis/figures as vector PDF.
%   Prerequisite:  python -m scripts.export_fig_data
%   Conventions:   Times New Roman, no titles, consistent sizing, vector output.
addpath(fileparts(mfilename('fullpath')));
figs = {@fig_design_coverage, @fig_measurement_validity, @fig_estimates, ...
        @fig_factor_matrix, @fig_representation, @fig_identifiability, ...
        @fig_threshold, @fig_blind, @fig_forward, @fig_modes};
ok = 0;
for k = 1:numel(figs)
    f = figs{k};
    try
        f(); ok = ok + 1;
    catch err
        fprintf(2, 'FAILED %s: %s\n', func2str(f), err.message);
    end
end
fprintf('rendered %d of %d figures\n', ok, numel(figs));
