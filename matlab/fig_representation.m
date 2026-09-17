function fig_representation()
%FIG_REPRESENTATION The angle failure is a representation defect, not a capacity one.
%   Panels (a) and (b): the same model, folds and hyperparameters under two target
%   parameterizations, with the influence-coefficient method as reference.
%   Panel (c): the size-matched control, in which both arms train on the same
%   number of conditions, so the degradation is attributable to angle novelty
%   rather than to reduced training data.
t = rk_load('representation_test.csv');
a = rk_load('angle_novelty_control.csv');
C = rk_colors();

protos   = {'condition','angle_sector','configuration'};
labels   = {'Condition','Angle sector','Configuration'};
variants = {'cartesian','equivariant','icm_prior'};
vnames   = {'Laboratory frame','Equivariant','Influence coefficient'};
vcolors  = {C.hybrid, C.pinn, C.physics};
cols     = {'r2_pred','phase_deg'};
ylabs    = {'Predictive {\itR}^2','Phase error (deg)'};

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');

for panel = 1:2
    ax = nexttile(tl); hold(ax,'on');
    Y = nan(numel(protos), numel(variants));
    for p = 1:numel(protos)
        for v = 1:numel(variants)
            r = t(strcmp(t.protocol,protos{p}) & strcmp(t.variant,variants{v}), :);
            if ~isempty(r), Y(p,v) = r.(cols{panel})(1); end
        end
    end
    b = bar(ax, Y, 'grouped', 'EdgeColor','none', 'BarWidth', 0.85);
    for v = 1:numel(variants), b(v).FaceColor = vcolors{v}; end
    if panel == 1
        yline(ax, 0, '-', 'Color', C.neutral, 'LineWidth', 0.75);
    end
    ylabel(ax, ylabs{panel});
    set(ax, 'XTick', 1:numel(protos), 'XTickLabel', labels, 'XTickLabelRotation', 25);
    grid(ax,'on'); rk_style(ax);
    if panel == 1
        lg = legend(ax, b, vnames, 'Location','southwest');
        set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box','off');
    end
end

ax = nexttile(tl); hold(ax,'on');
arms   = {'unseen_angle','size_matched_random'};
anames = {'Angle sector withheld','Random, size matched'};
acol   = {C.ml, C.neutral};
for k = 1:2
    s = a(strcmp(a.arm, arms{k}), :);
    n = height(s);
    if n > 1, x = k + linspace(-0.13, 0.13, n); else, x = k; end
    plot(ax, x, s.r2_pred, 'o', 'Color', acol{k}, 'MarkerFaceColor', acol{k}, ...
         'MarkerSize', 4, 'LineStyle','none');
    plot(ax, k + [-0.22 0.22], mean(s.r2_pred)*[1 1], '-', ...
         'Color', acol{k}, 'LineWidth', 1.6);
end
yline(ax, 0, '-', 'Color', C.neutral, 'LineWidth', 0.75);
set(ax, 'XTick', 1:2, 'XTickLabel', anames, 'XTickLabelRotation', 25, 'XLim', [0.5 2.5]);
ylabel(ax, 'Predictive {\itR}^2');
grid(ax,'on'); rk_style(ax);

rk_export(fig, 'fig_representation', 17.4, 7.2);
close(fig);
end
