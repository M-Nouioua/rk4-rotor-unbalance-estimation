function fig_forward()
%FIG_FORWARD Forward validation of the identified operator and fixed ICM.
%   The operator is fitted on training folds and predicts held-out conditions.
%   The ICM forward map uses its fixed trial-mass calibration. Panel (a) compares
%   predicted and measured response components. Panel (b) compares condition-level
%   relative-error distributions with measurement repeatability.
d = rk_load('pinn_forward.csv');
C = rk_colors();

meas = [];
pred = [];
icm = [];
for i = 0:7
    meas = [meas; 25.4 * d.(sprintf('meas_%d', i))];
    pred = [pred; 25.4 * d.(sprintf('pred_%d', i))];
    icm = [icm; 25.4 * d.(sprintf('icm_pred_%d', i))];
end

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

ax = nexttile(tl); hold(ax,'on');
lim = max(abs([meas; pred; icm])) * 1.05;
plot(ax, [-lim lim], [-lim lim], '--', 'Color', C.neutral, 'LineWidth', 0.9);
hOp = plot(ax, meas, pred, 'o', 'MarkerSize', 2.5, 'Color', C.pinn, ...
     'MarkerFaceColor', C.pinn, 'LineStyle', 'none');
hIcm = plot(ax, meas, icm, 'd', 'MarkerSize', 2.5, 'Color', C.physics, ...
     'LineStyle', 'none');
ssOp = 1 - sum((pred - meas).^2) / sum((meas - mean(meas)).^2);
ssIcm = 1 - sum((icm - meas).^2) / sum((meas - mean(meas)).^2);
xlabel(ax, ['Measured response component ' char(40) char(181) 'm' char(41)], 'Interpreter', 'none');
ylabel(ax, ['Predicted response component ' char(40) char(181) 'm' char(41)], 'Interpreter', 'none');
xlim(ax, [-lim lim]); ylim(ax, [-lim lim]);
text(ax, -0.95*lim, 0.9*lim, sprintf('Operator R^2 = %.3f\nICM R^2 = %.3f', ssOp, ssIcm), ...
     'FontName', rk_font(), 'FontSize', 8, 'VerticalAlignment', 'top');
legend(ax, [hOp hIcm], {'Physics-informed operator','Influence coefficient'}, ...
       'Location','southeast', 'Box','off');
grid(ax, 'on'); rk_style(ax);

ax = nexttile(tl); hold(ax,'on');
cids = unique(d.condition_id);
relOp = nan(numel(cids), 1);
relIcm = nan(numel(cids), 1);
for k = 1:numel(cids)
    s = d(strcmp(d.condition_id, cids{k}), :);
    M = zeros(height(s), 8); P = zeros(height(s), 8); I = zeros(height(s), 8);
    for i = 0:7
        M(:, i+1) = 25.4 * s.(sprintf('meas_%d', i));
        P(:, i+1) = 25.4 * s.(sprintf('pred_%d', i));
        I(:, i+1) = 25.4 * s.(sprintf('icm_pred_%d', i));
    end
    relOp(k) = median(vecnorm(P - M, 2, 2) ./ max(vecnorm(M, 2, 2), 1e-12));
    relIcm(k) = median(vecnorm(I - M, 2, 2) ./ max(vecnorm(M, 2, 2), 1e-12));
end
edgeMax = max([relOp; relIcm]);
edgeMax = max(0.05, 0.05 * ceil(edgeMax / 0.05));
edges = 0:0.05:(edgeMax + 0.05);
hOp = histogram(ax, relOp, 'BinEdges', edges, 'FaceColor', C.pinn, ...
                'FaceAlpha', 0.55, 'EdgeColor', 'none');
hIcm = histogram(ax, relIcm, 'BinEdges', edges, 'DisplayStyle', 'stairs', ...
                 'EdgeColor', C.physics, 'LineWidth', 1.2);
xline(ax, 0.011, '--', 'Color', C.neutral, 'LineWidth', 1.1);
xlabel(ax, 'Per-condition relative response error');
ylabel(ax, 'Conditions');
% The label used to sit inside the tallest bins. The limit is raised so the
% label has a clear strip above the histogram.
nmax = max([histcounts(relOp, edges), histcounts(relIcm, edges)]);
ylim(ax, [0 nmax*1.22]);
text(ax, 0.02, nmax*1.20, 'repeatability', ...
     'FontName', rk_font(), 'FontSize', 7.5, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
legend(ax, [hOp hIcm], {'Physics-informed operator','Influence coefficient'}, ...
       'Location','northeast', 'Box','off');
grid(ax, 'on'); rk_style(ax);

rk_export(fig, 'fig_forward', 17.4, 6.8);
close(fig);
end
