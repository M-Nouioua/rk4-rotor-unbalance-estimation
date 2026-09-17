function t = rk_load(name)
%RK_LOAD Read one exported table from analysis/fig_data.
%   Produced by `python -m scripts.export_fig_data`.
here = fileparts(mfilename('fullpath'));
f = fullfile(here, '..', 'analysis', 'fig_data', name);
if ~exist(f, 'file')
    error('rk_load:missing', ...
        '%s not found. Run: python -m scripts.export_fig_data', f);
end
t = readtable(f, 'VariableNamingRule', 'preserve');
end
