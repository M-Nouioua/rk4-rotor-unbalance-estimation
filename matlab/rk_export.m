function rk_export(fig, name, widthCm, heightCm)
%RK_EXPORT Export a figure as a vector PDF at an exact physical size.
%   Consistent dimensions across the paper. Vector output so the figure scales
%   without loss at any print resolution.
if nargin < 3 || isempty(widthCm),  widthCm  = 8.4;  end   % single column
if nargin < 4 || isempty(heightCm), heightCm = 6.2;  end
set(fig, 'Units', 'centimeters', ...
         'Position', [2 2 widthCm heightCm], ...
         'PaperUnits', 'centimeters', ...
         'PaperSize', [widthCm heightCm], ...
         'PaperPositionMode', 'auto', ...
         'Color', 'w', ...
         'InvertHardcopy', 'off');
outDir = fullfile(fileparts(mfilename('fullpath')), '..', 'analysis', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end
f = fullfile(outDir, [name '.pdf']);
if exist('exportgraphics', 'file')
    exportgraphics(fig, f, 'ContentType', 'vector', 'BackgroundColor', 'white');
else
    print(fig, f, '-dpdf', '-painters', '-r600');   % older MATLAB fallback
end
fprintf('wrote %s (%.1f x %.1f cm, vector)\n', f, widthCm, heightCm);
end
