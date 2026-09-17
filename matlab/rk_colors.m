function c = rk_colors()
%RK_COLORS Shared, colour-blind-safe palette. One colour per estimator family,
%   used identically in every figure so a reader learns the mapping once.
c = struct( ...
    'physics', [0.00 0.35 0.61], ...   % blue
    'ml',      [0.83 0.37 0.00], ...   % orange
    'hybrid',  [0.47 0.13 0.53], ...   % purple
    'pinn',    [0.00 0.50 0.35], ...   % green
    'neutral', [0.40 0.40 0.40], ...
    'light',   [0.75 0.75 0.75]);
end
