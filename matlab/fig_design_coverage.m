function fig_design_coverage()
%FIG_DESIGN_COVERAGE What the campaign covers, and where it does not.
%   Panel (a) every loaded disk point in magnitude and angle, by configuration.
%   Panel (b) the number of conditions per angle above 24 g.mm, which is the
%   coverage that limits any phase or anisotropy claim.
d = rk_load('design_coverage.csv');
C = rk_colors();
fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

ax = nexttile(tl); hold(ax,'on');
L = d(d.loaded == 1, :);
cfgs = unique(L.config);
% The table stores short configuration keys. The legend uses the names the
% manuscript uses in Section 2.6, so figure and text agree.
pretty = containers.Map( ...
    {'D1', 'D2', 'inphase', 'antiphase', 'baseline'}, ...
    {'Disk 1 alone', 'Disk 2 alone', 'In-phase', 'Anti-phase', 'Balanced'});
cfgnames = cell(1, numel(cfgs));
for k = 1:numel(cfgs)
    if isKey(pretty, cfgs{k})
        cfgnames{k} = pretty(cfgs{k});
    else
        cfgnames{k} = cfgs{k};
    end
end
cols = {C.physics, C.ml, C.hybrid, C.pinn, C.neutral};
h = gobjects(1, numel(cfgs));
for k = 1:numel(cfgs)
    s = L(strcmp(L.config, cfgs{k}), :);
    cc = cols{min(k, numel(cols))};
    h(k) = plot(ax, s.angle_deg, s.U_gmm, 'o', 'MarkerSize', 4, ...
                'Color', cc, 'MarkerFaceColor', cc, 'LineStyle', 'none');
end
yline(ax, 24, '--', 'Color', C.neutral, 'LineWidth', 0.9);
xlabel(ax, 'Unbalance angle (deg)');
ylabel(ax, 'Unbalance magnitude (g mm)');
set(ax, 'XTick', -180:45:180, 'XLim', [-200 200]);
grid(ax, 'on'); rk_style(ax);
lg = legend(ax, h, cfgnames, 'Location', 'northwest');
set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box', 'off');

ax = nexttile(tl); hold(ax,'on');
a = rk_load('design_angles_highU.csv');
ang = a{:, 1};
bar(ax, ang, a.n_conditions, 'FaceColor', C.pinn, 'EdgeColor', 'none', ...
    'BarWidth', 0.5);
xlabel(ax, 'Unbalance angle (deg)');
ylabel(ax, 'Conditions with U of 24 g mm or more');
set(ax, 'XTick', -180:45:180, 'XLim', [-200 200]);
grid(ax, 'on'); rk_style(ax);

rk_export(fig, 'fig_design_coverage', 17.4, 7.0);
close(fig);
end
