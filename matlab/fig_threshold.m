function fig_threshold()
%FIG_THRESHOLD Detection and localization depend on the decision threshold.
%   Panel (a) plane-localization accuracy against threshold, showing that the
%   ranking between estimator families inverts rather than being a fixed property.
%   Panel (b) sensitivity (solid) and specificity (dashed). Panel (c)
%   threshold-free areas under the curve. ROC and precision-recall have different
%   no-skill references, so no shared horizontal baseline is drawn.
d = rk_load('threshold_sweep.csv');
a = rk_load('detection_auc.csv');
C = rk_colors();
methods = {'physics', 'ml', 'hybrid'};
mnames  = {'Influence coefficient', 'Tree ensemble', 'Physics-augmented'};
mcolors = {C.physics, C.ml, C.hybrid};

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');

ax = nexttile(tl); hold(ax,'on');
h = gobjects(1, numel(methods));
for k = 1:numel(methods)
    s = d(strcmp(d.method, methods{k}) & ~isnan(d.thresh_gmm), :);
    h(k) = plot(ax, s.thresh_gmm, 100*s.localization, '-o', ...
                'Color', mcolors{k}, 'MarkerFaceColor', mcolors{k}, ...
                'MarkerSize', 3, 'LineWidth', 1.1);
end
xline(ax, 6, ':', 'Color', C.neutral, 'LineWidth', 0.9);
xlabel(ax, 'Decision threshold (g mm)');
ylabel(ax, 'Localization accuracy (%)');
grid(ax, 'on'); rk_style(ax);
lg = legend(ax, h, mnames, 'Location', 'southeast');
set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box', 'off');

ax = nexttile(tl); hold(ax,'on');
for k = 1:numel(methods)
    s = d(strcmp(d.method, methods{k}) & ~isnan(d.thresh_gmm), :);
    plot(ax, s.thresh_gmm, 100*s.sens, '-',  'Color', mcolors{k}, 'LineWidth', 1.1);
    plot(ax, s.thresh_gmm, 100*s.spec, '--', 'Color', mcolors{k}, 'LineWidth', 1.1);
end
xline(ax, 6, ':', 'Color', C.neutral, 'LineWidth', 0.9);
xlabel(ax, 'Decision threshold (g mm)');
ylabel(ax, 'Sensitivity, specificity (%)');
grid(ax, 'on'); rk_style(ax);

ax = nexttile(tl); hold(ax,'on');
Y = nan(numel(methods), 2);
for k = 1:numel(methods)
    r = a(strcmp(a.method, methods{k}), :);
    if ~isempty(r)
        Y(k,:) = [r.roc_auc(1) r.pr_auc(1)];
    end
end
b = bar(ax, Y, 'grouped', 'EdgeColor', 'none', 'BarWidth', 0.85);
b(1).FaceColor = C.physics;
b(2).FaceColor = C.pinn;
set(ax, 'XTick', 1:numel(methods), ...
        'XTickLabel', {'Influence coeff.', 'Tree ensemble', 'Physics-aug.'}, ...
        'XTickLabelRotation', 25);
ylabel(ax, 'Area under curve');
% Every bar reaches 0.85 or higher, so a legend inside the axes overlaps the
% data. The limit is raised to leave a clear strip and the legend sits in it.
ylim(ax, [0 1.22]);
grid(ax, 'on'); rk_style(ax);
lg = legend(ax, b, {'ROC', 'Precision-recall'}, 'Location', 'north', ...
            'Orientation', 'horizontal');
set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box', 'off');

rk_export(fig, 'fig_threshold', 17.4, 6.6);
close(fig);
end
