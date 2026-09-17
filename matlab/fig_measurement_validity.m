function fig_measurement_validity()
%FIG_MEASUREMENT_VALIDITY Run-up identification and the conditioning of the
%   two-plane inverse at the dwell speeds.
%   Panel (a) 1X amplitude against speed, with the two dwell speeds marked.
%   Panel (b) 1X phase. Panel (c) influence-matrix condition number per speed.
d = rk_load('runup_bode.csv');
c = rk_load('conditioning.csv');
C = rk_colors();
fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');

ax = nexttile(tl); hold(ax,'on');
plot(ax, d.rpm, d.amp_1x_um, '-', 'Color', C.physics, 'LineWidth', 0.9);
xline(ax, 1005, ':', 'Color', C.neutral, 'LineWidth', 0.9);
xline(ax, 1206, ':', 'Color', C.neutral, 'LineWidth', 0.9);
xlabel(ax, 'Shaft speed (rpm)');
ylabel(ax, ['1X amplitude ' char(40) char(181) 'm' char(41)], 'Interpreter', 'none');
xlim(ax, [1000 3000]);
grid(ax, 'on'); rk_style(ax);

ax = nexttile(tl); hold(ax,'on');
plot(ax, d.rpm, d.phase_1x_deg, '-', 'Color', C.physics, 'LineWidth', 0.9);
xline(ax, 1005, ':', 'Color', C.neutral, 'LineWidth', 0.9);
xline(ax, 1206, ':', 'Color', C.neutral, 'LineWidth', 0.9);
xlabel(ax, 'Shaft speed (rpm)');
ylabel(ax, '1X phase (deg)');
xlim(ax, [1000 3000]); ylim(ax, [-180 180]);
set(ax, 'YTick', -180:90:180);
grid(ax, 'on'); rk_style(ax);

ax = nexttile(tl); hold(ax,'on');
% The exported table keys the dwell speeds as N1 and N2. Those labels mean
% nothing to a reader, so the bars are named by the measured speed instead.
sp = strrep(strrep(cellstr(string(c.speed)), 'N1', '1005 rpm'), 'N2', '1206 rpm');
bar(ax, categorical(sp, sp), c.cond_A, 'FaceColor', C.hybrid, ...
    'EdgeColor', 'none', 'BarWidth', 0.5);
ylabel(ax, 'Influence-matrix condition number');
grid(ax, 'on'); rk_style(ax);

rk_export(fig, 'fig_measurement_validity', 17.4, 6.4);
close(fig);
end
