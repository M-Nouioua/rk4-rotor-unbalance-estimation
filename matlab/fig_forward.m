function fig_forward()
%FIG_FORWARD Forward validation of the identified operator, which is the claim
%   that distinguishes a twin from a regression.
%   The operator is fitted on held-out folds and then used to predict the measured
%   1X response of conditions it was not fitted to, taking the known unbalance as
%   an input. The tree estimators cannot be tested this way, because they only
%   invert. Panel (a) predicted against measured response components with the
%   equality line. Panel (b) the distribution of per-condition relative error,
%   shown against the back-to-back repeatability of the measurement, so that
%   systematic model-form error is separated from measurement noise.
d = rk_load('pinn_forward.csv');
C = rk_colors();

meas = [];
pred = [];
for i = 0:7
    meas = [meas; 25.4 * d.(sprintf('meas_%d', i))];
    pred = [pred; 25.4 * d.(sprintf('pred_%d', i))];
end

fig = figure('Visible','off');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

ax = nexttile(tl); hold(ax,'on');
lim = max(abs([meas; pred])) * 1.05;
plot(ax, [-lim lim], [-lim lim], '--', 'Color', C.neutral, 'LineWidth', 0.9);
plot(ax, meas, pred, 'o', 'MarkerSize', 2.5, 'Color', C.pinn, ...
     'MarkerFaceColor', C.pinn, 'LineStyle', 'none');
ss = 1 - sum((pred - meas).^2) / sum((meas - mean(meas)).^2);
xlabel(ax, ['Measured response component ' char(40) char(181) 'm' char(41)], 'Interpreter', 'none');
ylabel(ax, ['Predicted response component ' char(40) char(181) 'm' char(41)], 'Interpreter', 'none');
xlim(ax, [-lim lim]); ylim(ax, [-lim lim]);
text(ax, -0.95*lim, 0.9*lim, sprintf('R^2 = %.3f', ss), ...
     'FontName', rk_font(), 'FontSize', 8, 'VerticalAlignment', 'top');
grid(ax, 'on'); rk_style(ax);

ax = nexttile(tl); hold(ax,'on');
cids = unique(d.condition_id);
rel = nan(numel(cids), 1);
for k = 1:numel(cids)
    s = d(strcmp(d.condition_id, cids{k}), :);
    M = zeros(height(s), 8); P = zeros(height(s), 8);
    for i = 0:7
        M(:, i+1) = 25.4 * s.(sprintf('meas_%d', i));
        P(:, i+1) = 25.4 * s.(sprintf('pred_%d', i));
    end
    rel(k) = median(vecnorm(P - M, 2, 2) ./ max(vecnorm(M, 2, 2), 1e-12));
end
histogram(ax, rel, 'BinWidth', 0.05, 'FaceColor', C.pinn, 'EdgeColor', 'none');
xline(ax, 0.011, '--', 'Color', C.physics, 'LineWidth', 1.1);
xlabel(ax, 'Per-condition relative response error');
ylabel(ax, 'Conditions');
% The label used to sit inside the tallest bins. The limit is raised so the
% label has a clear strip above the histogram.
nmax = max(histcounts(rel, 'BinWidth', 0.05));
ylim(ax, [0 nmax*1.22]);
text(ax, 0.02, nmax*1.20, 'repeatability', ...
     'FontName', rk_font(), 'FontSize', 7.5, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
grid(ax, 'on'); rk_style(ax);

rk_export(fig, 'fig_forward', 17.4, 6.8);
close(fig);
end
