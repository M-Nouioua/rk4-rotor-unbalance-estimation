function fig_factor_matrix()
%FIG_FACTOR_MATRIX Estimator performance across evaluation protocols.
%   Panel (a) predictive R^2, panel (b) circular phase error above 24 g.mm.
%   The figure shows that the ranking depends on which physical factor is
%   withheld, so the protocol axis is ordered by deployment question rather than
%   by difficulty. The influence-coefficient coefficients are fixed, but its
%   metrics are evaluated on each protocol's own test subset.
t = rk_load('factor_matrix.csv');
C = rk_colors();

protos = {'random_acquisition','condition','magnitude_interpolating', ...
          'magnitude_extrapolating','configuration','angle_sector'};
labels = {'Random','Condition','Magnitude (interp.)','Magnitude (extrap.)', ...
          'Configuration','Angle sector'};
methods = {'ml','hybrid','pinn_iso','physics'};
mnames  = {'Tree ensemble','Physics-augmented','Physics-informed (equivariant)', ...
           'Influence coefficient'};
mcolors = {C.ml, C.hybrid, C.pinn, C.physics};
markers = {'o','s','^','d'};
styles  = {'-','-','-','--'};
cols    = {'r2_pred','phase_deg'};
ylabs   = {'Predictive {\itR}^2','Phase error (deg)'};

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');
hLeg = gobjects(0); nLeg = {};

for panel = 1:2
    ax = nexttile(tl); hold(ax,'on');
    if panel == 1
        yline(ax, 0, '-', 'Color', C.light, 'LineWidth', 0.75);
    end
    hM = gobjects(1, numel(methods));
    for m = 1:numel(methods)
        y = nan(1, numel(protos));
        for p = 1:numel(protos)
            r = t(strcmp(t.method, methods{m}) & strcmp(t.protocol, protos{p}), :);
            if ~isempty(r), y(p) = r.(cols{panel})(1); end
        end
        hM(m) = plot(ax, 1:numel(protos), y, [styles{m} markers{m}], ...
                     'Color', mcolors{m}, 'MarkerFaceColor', mcolors{m}, ...
                     'MarkerSize', 4, 'LineWidth', 1.1);
    end
    set(ax, 'XTick', 1:numel(protos), 'XTickLabel', labels, ...
            'XTickLabelRotation', 38, 'XLim', [0.7 numel(protos)+0.3]);
    ylabel(ax, ylabs{panel});
    grid(ax,'on'); rk_style(ax);
    if panel == 1
        hLeg = hM;
        nLeg = mnames;
        lg = legend(ax, hLeg, nLeg, 'Location','southwest');
        set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box','off');
    end
end

rk_export(fig, 'fig_factor_matrix', 17.4, 7.6);
close(fig);
end
