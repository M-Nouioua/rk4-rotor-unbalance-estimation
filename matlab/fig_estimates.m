function fig_estimates()
%FIG_ESTIMATES Condition-level estimated against known unbalance, per estimator,
%   under condition-wise cross-validation. The dashed line is equality, and the
%   solid line is the fitted calibration; departure of its slope from unity is
%   shrinkage.
d = rk_load('estimates_condition.csv');
C = rk_colors();
methods = {'physics', 'ml', 'hybrid'};
mnames  = {'Influence coefficient', 'Tree ensemble', 'Physics-augmented'};
mcolors = {C.physics, C.ml, C.hybrid};

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');
for k = 1:numel(methods)
    ax = nexttile(tl); hold(ax,'on');
    s = d(strcmp(d.method, methods{k}) & d.loaded == 1, :);
    plot(ax, [0 65], [0 65], '--', 'Color', C.neutral, 'LineWidth', 0.9);
    plot(ax, s.true_mag, s.est_mag, 'o', 'MarkerSize', 3.5, ...
         'Color', mcolors{k}, 'MarkerFaceColor', mcolors{k}, 'LineStyle', 'none');
    p = polyfit(s.true_mag, s.est_mag, 1);
    plot(ax, [0 65], polyval(p, [0 65]), '-', 'Color', mcolors{k}, 'LineWidth', 1.1);
    xlabel(ax, 'Applied unbalance (g mm)');
    if k == 1
        ylabel(ax, 'Estimated unbalance (g mm)');
    end
    xlim(ax, [0 65]); ylim(ax, [0 80]);
    txt = sprintf('%s\nslope %.2f', mnames{k}, p(1));
    text(ax, 2, 77, txt, 'FontName', rk_font(), 'FontSize', 7.5, ...
         'VerticalAlignment', 'top');
    grid(ax, 'on'); rk_style(ax);
end
rk_export(fig, 'fig_estimates', 17.4, 6.6);
close(fig);
end
