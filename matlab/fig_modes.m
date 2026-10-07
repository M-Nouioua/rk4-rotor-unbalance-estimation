function fig_modes()
%FIG_MODES Inverse and forward accuracy against the number of modes.
%   The two directions disagree at one mode and agree from two onwards. A single
%   mode contributes a rank-one residue, so the two disks' columns are
%   proportional and the inverse cannot separate the planes, while the forward
%   direction is unaffected. That asymmetry is the argument for validating a twin
%   in both directions rather than only one.
%   Three random restarts per fit, selected on the training loss.
%   A single y axis per panel is used deliberately: yyaxis conflicts with the
%   shared axis styling, and a second scale here would add no information.
d = rk_load('pinn_modes.csv');
C = rk_colors();

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

ax = nexttile(tl); hold(ax,'on');
yline(ax, 0, '-', 'Color', C.light, 'LineWidth', 0.75);
h1 = plot(ax, d.modes, d.inv_r2_pred, '-o', 'Color', C.pinn, ...
          'MarkerFaceColor', C.pinn, 'MarkerSize', 4, 'LineWidth', 1.1);
h2 = plot(ax, d.modes, d.fwd_r2_pred, '-s', 'Color', C.physics, ...
          'MarkerFaceColor', C.physics, 'MarkerSize', 4, 'LineWidth', 1.1);
xlabel(ax, 'Number of modes');
ylabel(ax, 'Predictive {\itR}^2');
set(ax, 'XTick', d.modes);
grid(ax, 'on'); rk_style(ax);
lg = legend(ax, [h1 h2], {'Inverse (unbalance)', 'Forward (response)'}, ...
            'Location', 'southeast');
set(lg, 'FontName', rk_font(), 'FontSize', 7.5, 'Box', 'off');

% An inset was tried here to expand the two-to-six-mode range, where every score
% sits between 0.93 and 0.96. Inside a tiled layout the inset could not be
% placed reliably and it overlaid the panel, so the full range is kept: it shows
% the single-mode inverse failure honestly, and Table 8 carries the values that
% the compressed upper region cannot resolve.

ax = nexttile(tl); hold(ax,'on');
plot(ax, d.modes, d.fwd_rel_error, '-o', 'Color', C.pinn, ...
     'MarkerFaceColor', C.pinn, 'MarkerSize', 4, 'LineWidth', 1.1);
yline(ax, 0.011, '--', 'Color', C.physics, 'LineWidth', 1.1);
% The end labels are aligned inwards so neither runs off an edge, the axis is
% padded so the first is not clipped by the limit, and every label clears the
% curve by a fixed offset rather than landing on it.
for k = 1:height(d)
    if k == 1
        align = 'left';
    elseif k == height(d)
        align = 'right';
    else
        align = 'center';
    end
    text(ax, d.modes(k), d.fwd_rel_error(k) + 0.030, ...
         sprintf('%d par.', d.n_parameters(k)), ...
         'FontName', rk_font(), 'FontSize', 7, ...
         'HorizontalAlignment', align, 'VerticalAlignment', 'bottom');
end
xlim(ax, [min(d.modes) - 0.35, max(d.modes) + 0.35]);
xlabel(ax, 'Number of modes');
ylabel(ax, 'Relative response error');
set(ax, 'XTick', d.modes);
ylim(ax, [0 max(d.fwd_rel_error)*1.35]);
grid(ax, 'on'); rk_style(ax);
text(ax, max(d.modes)*0.72, 0.028, 'measurement repeatability', ...
     'FontName', rk_font(), 'FontSize', 7, 'Color', C.physics);

rk_export(fig, 'fig_modes', 17.4, 6.8);
close(fig);
end
