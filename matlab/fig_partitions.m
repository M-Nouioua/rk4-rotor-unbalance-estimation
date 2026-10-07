function fig_partitions()
%FIG_PARTITIONS What each evaluation protocol withholds, drawn on the design.
%   The six protocols are the central claim of the paper and are otherwise
%   carried by prose alone. Each panel shows the whole labelled design once:
%   columns are the eight 45 degree angle sectors, each subdivided into a fixed
%   slot per loading configuration, and rows are the total applied unbalance
%   levels. A cell is filled when the campaign holds that combination, and
%   coloured by its role in one representative fold.
%
%   Giving every configuration the same slot in every cell makes the
%   configuration protocol read as a vertical stripe, and the angular protocol
%   as a column, so the reader sees what is withheld rather than reading it.
%
%   The record-level control is the one protocol that splits a condition across
%   both sides, and those conditions are drawn in a third colour: that leakage
%   is the reason the control is reported at all.
d = rk_load('partition_map.csv');
C = rk_colors();

protos = {'Random records', 'Condition', 'Magnitude, interpolating', ...
          'Magnitude, extrapolating', 'Configuration', 'Angle sector'};
cfgs   = {'D1', 'D2', 'inphase', 'antiphase', 'baseline'};
cfgLab = {'D1', 'D2', 'in', 'anti', 'bal'};

levels = unique(d.total_gmm);
levels = sort(levels);
nL = numel(levels);
nS = 8;
nC = numel(cfgs);

col = struct('train', [0.80 0.89 0.80], ...      % muted green
             'test',  [0.76 0.85 0.93], ...      % muted blue
             'split', [0.95 0.78 0.70], ...      % muted red, the leaky control
             'calibration', [0.88 0.88 0.88]);

fig = figure('Visible','off');
tl = tiledlayout(fig, 3, 2, 'TileSpacing','compact', 'Padding','compact');

for p = 1:numel(protos)
    ax = nexttile(tl); hold(ax,'on');
    s = d(strcmp(d.protocol_label, protos{p}), :);

    % faint backdrop marking every combination the campaign could hold
    for ii = 1:nS
        for jj = 1:nL
            rectangle(ax, 'Position', [ii-0.5, jj-0.5, 1, 1], ...
                      'FaceColor', 'none', 'EdgeColor', [0.93 0.93 0.93], ...
                      'LineWidth', 0.3);
        end
    end

    w = 1 / (nC + 0.5);
    for k = 1:height(s)
        ci = find(strcmp(cfgs, s.config{k}), 1);
        if isempty(ci), continue; end
        li = find(levels == s.total_gmm(k), 1);
        x = (s.sector(k) + 1) - 0.5 + (ci - 1) * w + 0.25 * w;
        rectangle(ax, 'Position', [x, li - 0.42, w * 0.9, 0.84], ...
                  'FaceColor', col.(s.role{k}), 'EdgeColor', 'none');
    end

    set(ax, 'XLim', [0.5 nS+0.5], 'YLim', [0.5 nL+0.5], ...
            'XTick', 1:nS, 'XTickLabel', compose('%d', (0:nS-1)*45), ...
            'YTick', 1:nL, 'YTickLabel', compose('%g', levels), ...
            'TickLength', [0 0]);
    if p > 4, xlabel(ax, 'Applied angle (deg)'); end
    if mod(p, 2) == 1, ylabel(ax, 'Total unbalance (g mm)'); end
    nf = s.n_folds(1);
    text(ax, 0.6, nL + 0.30, sprintf('%s   (fold 1 of %d)', protos{p}, nf), ...
         'FontName', rk_font(), 'FontSize', 8.5, 'VerticalAlignment', 'bottom');
    rk_style(ax);
    set(ax, 'Box', 'on', 'XColor', [0.45 0.45 0.45], 'YColor', [0.45 0.45 0.45]);
end

% one shared legend, drawn as proxy patches so it names the roles once
h = gobjects(1, 4);
names = {'Trained on', 'Held out and scored', ...
         'Split across both, leakage', 'Calibration, never scored'};
keys = {'train', 'test', 'split', 'calibration'};
for k = 1:4
    h(k) = patch(ax, NaN, NaN, col.(keys{k}), 'EdgeColor', 'none');
end
lg = legend(ax, h, names, 'Orientation', 'horizontal', 'NumColumns', 4);
lg.Layout.Tile = 'south';
set(lg, 'FontName', rk_font(), 'FontSize', 8, 'Box', 'off');

rk_export(fig, 'fig_partitions', 17.4, 15.0);
close(fig);
end
