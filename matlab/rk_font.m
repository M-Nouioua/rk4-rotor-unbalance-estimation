function name = rk_font()
%RK_FONT Manuscript font, with a graceful fallback if it is unavailable.
name = 'Times New Roman';
persistent warned
if isempty(warned)
    warned = true;
    if ~any(strcmpi(listfonts(), name))
        warning('rk_font:missing', ...
            ['Times New Roman not found on this system. Figures will not match ' ...
             'the required manuscript font. Install it before exporting finals.']);
    end
end
end
