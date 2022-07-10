function out = is_valid_box(box)
% Check box is valid or not
    xl = box(1);
    yl = box(2);
    xr = box(3);
    yr = box(4);

    w = xr - xl +1;
    h = yr - yl +1;

    % drop too small or out of image boxes 
    if min(w,h) < 20 || xl < 0 || yl <0
        out = false;
    else
        out = true;
    end
end