function fig_factor_matrix()
%FIG_FACTOR_MATRIX Estimator performance across evaluation protocols.
%   Panel (a) predictive R^2, panel (b) circular phase error above 24 g.mm.
%   The figure shows that the ranking depends on which physical factor is
%   withheld, so the protocol axis is ordered by deployment question rather than
%   by difficulty. The influence-coefficient method is fitted only from the fixed
%   calibration conditions, so it does not depend on the split and appears as a
%   horizontal reference.
t = rk_load('factor_matrix.csv');
C = rk_colors();

protos = {'random_acquisition','condition','magnitude_interpolating', ...
          'magnitude_extrapolating','configuration','angle_sector'};
labels = {'Random','Condition','Magnitude (interp.)','Magnitude (extrap.)', ...
          'Configuration','Angle sector'};
methods = {'ml','hybrid','pinn_iso'};
mnames  = {'Tree ensemble','Physics-augmented','Physics-informed (equivariant)'};
mcolors = {C.ml, C.hybrid, C.pinn};
markers = {'o','s','^'};
cols    = {'r2_pred','phase_deg'};
ylabs   = {'Predictive {\itR}^2','Phase error (deg)'};

pr = t(strcmp(t.method,'physics'), :);
icm = [NaN NaN];
if ~isempty(pr), icm = [pr.r2_pred(1) pr.phase_deg(1)]; end

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');
hLeg = gobjects(0); nLeg = {};

for panel = 1:2
    ax = nexttile(tl); hold(ax,'on');
    if panel == 1
        yline(ax, 0, '-', 'Color', C.light, 'LineWidth', 0.75);
    end
    hRef = gobjects(0);
    if ~isnan(icm(panel))
        hRef = yline(ax, icm(panel), '--', 'Color', C.physics, 'LineWidth', 1.0);
    end
    hM = gobjects(1, numel(methods));
    for m = 1:numel(methods)
        y = nan(1, numel(protos));
        for p = 1:numel(protos)
            r = t(strcmp(t.method, methods{m}) & strcmp(t.protocol, protos{p}), :);
            if ~isempty(r), y(p) = r.(cols{panel})(1); end
        end
        hM(m) = plot(ax, 1:numel(protos), y, ['-' markers{m}], ...
                     'Color', mcolors{m}, 'MarkerFaceColor', mcolors{m}, ...
                     'MarkerSize', 4, 'LineWidth', 1.1);
    end
    set(ax, 'XTick', 1:numel(protos), 'XTickLabel', labels, ...
            'XTickLabelRotation', 38, 'XLim', [0.7 numel(protos)+0.3]);
    ylabel(ax, ylabs{panel});
    grid(ax,'on'); rk_style(ax);
    if panel == 1
        hLeg = [hM, hRef];
        nLeg = mnames;
        if ~isempty(hRef), nLeg = [mnames, {'Influence coefficient'}]; end
        lg = legend(ax, hLeg, nLeg, 'Location','southwest');
        set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box','off');
    end
end

rk_export(fig, 'fig_factor_matrix', 17.4, 7.6);
close(fig);
end
