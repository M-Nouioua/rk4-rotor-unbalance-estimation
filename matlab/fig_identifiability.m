function fig_identifiability()
%FIG_IDENTIFIABILITY Physically correct structure that the design cannot identify
%   degrades generalization.
%   Both variants share the same modal operator. The isotropic variant is
%   exactly equivariant and uses 28 active parameters. Adding backward-whirl
%   residues doubles the parameter count and improves the in-distribution fit,
%   but the campaign contains only four distinct unbalance angles above
%   24 g.mm, so those residues are poorly constrained and the model loses its
%   equivariance. Panel (a) predictive R^2, panel (b) phase error.
t = rk_load('factor_matrix.csv');
C = rk_colors();

protos = {'condition','magnitude_interpolating','magnitude_extrapolating', ...
          'configuration','angle_sector'};
labels = {'Condition','Magnitude (interp.)','Magnitude (extrap.)', ...
          'Configuration','Angle sector'};
variants = {'pinn_iso','pinn_aniso'};
vnames   = {'Isotropic (equivariant, 42 par.)', ...
            'Anisotropic (78 par.)'};
vcolors  = {C.pinn, C.ml};
cols     = {'r2_pred','phase_deg'};
ylabs    = {'Predictive {\itR}^2','Phase error (deg)'};

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

for panel = 1:2
    ax = nexttile(tl); hold(ax,'on');
    Y = nan(numel(protos), numel(variants));
    for p = 1:numel(protos)
        for v = 1:numel(variants)
            r = t(strcmp(t.method,variants{v}) & strcmp(t.protocol,protos{p}), :);
            if ~isempty(r), Y(p,v) = r.(cols{panel})(1); end
        end
    end
    b = bar(ax, Y, 'grouped', 'EdgeColor','none', 'BarWidth', 0.88);
    for v = 1:numel(variants), b(v).FaceColor = vcolors{v}; end
    if panel == 1
        yline(ax, 0, '-', 'Color', C.neutral, 'LineWidth', 0.75);
    end
    ylabel(ax, ylabs{panel});
    set(ax, 'XTick', 1:numel(protos), 'XTickLabel', labels, 'XTickLabelRotation', 32);
    grid(ax,'on'); rk_style(ax);
    if panel == 1
        lg = legend(ax, b, vnames, 'Location','southwest');
        set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box','off');
    end
end

rk_export(fig, 'fig_identifiability', 17.4, 7.4);
close(fig);
end
