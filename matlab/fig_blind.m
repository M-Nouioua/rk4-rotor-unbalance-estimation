function fig_blind()
%FIG_BLIND Blind validation on ten conditions whose labels were revealed only
%   after prediction.
%   Panel (a) estimated against known magnitude for the classical, tree and
%   physics-informed estimators, with balanced cases at the origin. Panel (b)
%   loaded-point magnitude error. Panel (c) phase error above 24 g.mm.
%   Ten independent conditions only, so the panels show every point rather than
%   a summary statistic.
d = rk_load('blind_points.csv');
C = rk_colors();
try
    p = rk_load('pinn_blind.csv');
    p = p(strcmp(p.variant, 'iso'), :);
    pin = table(repmat({'pinn'}, height(p), 1), p.condition_id, p.disk, ...
                p.true_gmm, p.est_gmm, p.true_ang, p.est_ang, ...
                abs(mod(p.est_ang - p.true_ang + 180, 360) - 180), ...
                p.phase_label_valid, p.balanced, ...
                'VariableNames', d.Properties.VariableNames);
    d = [d; pin];
catch err
    % Report why the operator series is missing. A bare "not found" message
    % here previously hid a column-count mismatch, and the figure was published
    % silently without the operator.
    warning('fig_blind:nopinn', ...
            'operator series omitted: %s', err.message);
end

methods = {'physics', 'hybrid', 'pinn'};
mnames  = {'Influence coefficient', 'Physics-augmented', 'Physics-informed'};
mcolors = {C.physics, C.hybrid, C.pinn};
markers = {'o', 's', '^'};

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');

ax = nexttile(tl); hold(ax,'on');
plot(ax, [0 55], [0 55], '--', 'Color', C.neutral, 'LineWidth', 0.9);
h = gobjects(0); nm = {};
for k = 1:numel(methods)
    s = d(strcmp(d.method, methods{k}), :);
    if isempty(s), continue; end
    h(end+1) = plot(ax, s.true_gmm, s.est_gmm, markers{k}, 'MarkerSize', 4.5, ...
                    'Color', mcolors{k}, 'MarkerFaceColor', mcolors{k}, ...
                    'LineStyle', 'none');
    nm{end+1} = mnames{k};
end
xlabel(ax, 'Applied unbalance (g mm)');
ylabel(ax, 'Estimated unbalance (g mm)');
xlim(ax, [-2 55]); ylim(ax, [-2 70]);
grid(ax, 'on'); rk_style(ax);
lg = legend(ax, h, nm, 'Location', 'northwest');
set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box', 'off');

ax = nexttile(tl); hold(ax,'on');
for k = 1:numel(methods)
    s = d(strcmp(d.method, methods{k}) & d.balanced == 0, :);
    if isempty(s), continue; end
    e = abs(s.est_gmm - s.true_gmm);
    x = k + linspace(-0.16, 0.16, numel(e));
    plot(ax, x, e, markers{k}, 'MarkerSize', 4, 'Color', mcolors{k}, ...
         'MarkerFaceColor', mcolors{k}, 'LineStyle', 'none');
    plot(ax, k + [-0.26 0.26], median(e)*[1 1], '-', ...
         'Color', mcolors{k}, 'LineWidth', 1.6);
end
set(ax, 'XTick', 1:numel(methods), 'XTickLabel', {'Influence coeff.', 'Physics-aug.', 'Physics-inf.'}, ...
        'XTickLabelRotation', 25, 'XLim', [0.5 numel(methods)+0.5]);
ylabel(ax, 'Loaded-point magnitude error (g mm)');
grid(ax, 'on'); rk_style(ax);

ax = nexttile(tl); hold(ax,'on');
for k = 1:numel(methods)
    s = d(strcmp(d.method, methods{k}) & d.true_gmm >= 24, :);
    if isempty(s), continue; end
    e = s.ang_err(~isnan(s.ang_err));
    if isempty(e), continue; end
    x = k + linspace(-0.16, 0.16, numel(e));
    plot(ax, x, e, markers{k}, 'MarkerSize', 4, 'Color', mcolors{k}, ...
         'MarkerFaceColor', mcolors{k}, 'LineStyle', 'none');
    plot(ax, k + [-0.26 0.26], median(e)*[1 1], '-', ...
         'Color', mcolors{k}, 'LineWidth', 1.6);
end
set(ax, 'XTick', 1:numel(methods), 'XTickLabel', {'Influence coeff.', 'Physics-aug.', 'Physics-inf.'}, ...
        'XTickLabelRotation', 25, 'XLim', [0.5 numel(methods)+0.5]);
ylabel(ax, 'Phase error for U of 24 g mm or more (deg)');
grid(ax, 'on'); rk_style(ax);

rk_export(fig, 'fig_blind', 17.4, 6.8);
close(fig);
end
