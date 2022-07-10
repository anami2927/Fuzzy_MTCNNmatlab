function get_p_net_data(data_dir, p_net_size, output_files)
% Input: data directory, p_net size and 3 files including 
    % train_p_net_pos.txt
    % train_p_net_neg.txt
    % train_p_net_part.txt

    annotation_file = fullfile(data_dir,'wider_face_train.txt');
    image_dir = fullfile(data_dir,'/WIDER_train/images');
    save_dir = fullfile(data_dir,'p_net');

 % Create 3 files
    pos_file = fopen(output_files(1),"w");
    part_file = fopen(output_files(2),"w");
    neg_file = fopen(output_files(3),"w");

    total_box_num = 0;
    annotations = readlines(annotation_file);

    image_num = numel(annotations);

    for i=1:image_num
        line = annotations(i);
        line = strip(line);
        annotation = split(line);
        box_num_this_image = round(numel(annotation)/4);
        total_box_num = total_box_num + box_num_this_image;
    end
    msg = ["total image: ", image_num, 'total box num: ', total_box_num];
    disp(msg)

    % pos part neg number derived from one box
    pos_and_part_per_box = 20;
    neg_per_box = 5;
    neg_per_image = 50;
    neg_count = 0;
    part_count = 0;
    pos_count = 0;

    for i=1:image_num
        line = annotations(i);
        line = strip(line);
        annotation = split(line);
        image_path = annotation(1);
        image_path = strcat(image_path,".jpg");
        path_to_image = fullfile(image_dir,image_path);
        image = imread(path_to_image);  
        [height, width,~] = size(image);
        fprintf(pos_file,image_path+'\n');
        fprintf(part_file,image_path+'\n');
        fprintf(neg_file,image_path+'\n');
        boxes = str2double(annotation(2:end))';
        boxes = reshape(boxes,[numel(annotation(2:end))/4,4]);

        neg_amount = 0;

        while neg_amount <  neg_per_image
            % Sample some negative image
            min_value = min(height,width)/2;
            random_size = floor(p_net_size + (min_value-p_net_size).*rand(1,1));
    
            % Random top-left point
            xl = floor(0+ ((width-random_size-0)).*rand(1,1));
            yl = floor(0+ ((height-random_size-0)).*rand(1,1));
    
            % Crop box
            crop_box = [xl, yl, xl+random_size, yl+random_size];
            iou_value = iou(double(crop_box),double(boxes));
            if max(iou_value) < 0.3
                % neg if iou < 0.3
                data = num2str(crop_box);
                data = append(data,' 0 \n');
                fprintf(neg_file,data);
                neg_amount = neg_amount+1;
                neg_count = neg_count+1;
            end
        end % end negative part 

        % 5  neg part around the box
        [row, col, ch] = size(boxes);
        for j=1:row
            box = boxes(j,:);
            if ~   is_valid_box(box)
                continue;
            end

            xl = box(1);
            yl = box(2);
            xr = box(3);
            yr = box(4);

            w = xr-xl+1; % box width
            h = yr-yl+1; % box height

            for k=1:neg_per_box
                random_size = floor(p_net_size + (min_value-p_net_size).*rand(1,1));
                max_size1 = max(-random_size, -xl);
                max_size2 = max(-random_size, -yl);

                delta_x =   floor(max_size1 + (w-max_size1).*rand(1,1));
                delta_y =   floor(max_size2 + (h-max_size2).*rand(1,1));

                % top left corner
                xl_crop = round(max(0,xl+delta_x));
                yl_crop = round(max(0,yl+delta_y));

                % drop too large
                if xl_crop + random_size > width || yl_crop + random_size > height
                    continue
                end
                crop_box = [xl_crop yl_crop xl_crop+random_size yl_crop+random_size];
                data = num2str(crop_box);

                iou_value = iou(crop_box,boxes);
                if max(iou_value) < 0.3
                    data = append(data,' 0 ',' \n');
                    fprintf(neg_file,data);
                    neg_count = neg_count+1;
                end

            end % end for neg_around_box

            % Start for pos_part_around_box
            for l=1:pos_and_part_per_box
                min_w_h = 0.8*min(w,h);
                max_w_h = 1.25*max(w,h);

                % random number
                random_size = floor(min_w_h + (max_w_h-min_w_h).*rand(1,1));
                delta_x = floor(-w*0.2 + (w*0.2-(-w*0.2)).*rand(1,1));
                delta_y = floor(-h*0.2 + (h*0.2-(-h*0.2)).*rand(1,1));

                % offset the cropped box's center to the center of true box
                % along width and height 

                xl_crop = round(max(xl+ w/2 + delta_x - random_size /2, 0));
                yl_crop = round(max(yl+ h/2 + delta_y - random_size /2, 0));
                xr_crop = xl_crop + random_size -1;
                yr_crop = yl_crop + random_size -1; 

                if xr_crop > width || yr_crop > height
                    continue
                end

                crop_box = [xl_crop yl_crop xr_crop yr_crop];
                data = num2str(crop_box);

                iou_value = iou(double(crop_box),double(boxes));

                % Offset, relatively
                offset_xl = (xl-xl_crop)/random_size;
                offset_yl = (yl-yl_crop)/random_size;
                offset_xr = (xr-xr_crop)/random_size;
                offset_yr = (yr-yr_crop)/random_size;

                offsets = [offset_xl offset_yl offset_xr offset_yr];
                offsets = num2str(offsets);

                if max(iou_value) >=0.65 
                    data = append(data, ' 1 ', offsets, ' \n');
                    fprintf(pos_file,data);
                    pos_count = pos_count+1;
                elseif max(iou_value) >=0.4 
                    data = append(data, ' -1 ', offsets, ' \n');
                    fprintf(part_file,data);
                    part_count = part_count+1;
                end
            end
        end
    end
    fclose(pos_file);
         fclose(part_file);
         fclose(neg_file);
            % End for pos_part_around_box
   
end