function out = iou(box, boxes)
    if numel(boxes) == 0
        out = 0;
    end

    % area
    box_area = (box(3)-box(1)+1) * (box(4)-box(2)+1);

    % area
    boxes_areas = (boxes(:,3) - boxes(:,1) + 1) .* (boxes(:,4)-boxes(:,2)+1);

    % overlap part of top-left, bottom right point
    xl = max(box(1),boxes(:,1));
    yl = max(box(2),boxes(:,2));
    xr = min(box(3),boxes(:,3));
    yr = min(box(4),boxes(:,4));

    % overlap width
    width = max(0,xr-xl+1);
    height = max(0,yr-yl+1);
    area = width.*height;
    out = area./(box_area+boxes_areas - area +eps);

end