function data_example = get_dataset(data_dir,data_file, size, gray)
%   Get data from text file
%   Arg: Input 
%       param data file -> data_file
%       param size -> size
%       param gray flag -> gray
    annotation_file = fullfile(data_dir,data_file);
    images_file = fullfile("data/WIDER_train/images");
    image_list = fopen(annotation_file,"r");
    dataset = [];
    image = [];
    save_path = fullfile("data/p_net_12/");
    num_image = 1;
    count = 1;
    annotations = readlines(annotation_file);
    num_line = numel(annotations);

    for i=1:num_line-1
        line = annotations(i);
        line = char(line);
        if strcmp(line(end-3:end),'.jpg')
            image = imread(fullfile(images_file,line));
        else
            line = str2num(line);
            xl = line(1);
            yl = line(2);
            xr = line(3);
            yr = line(4);

            crop_image = imcrop(image,[xl yl xr-xl yr-yl]);
            crop_image = imresize3(crop_image, [12 12 3]);

%             folder = 'data/p_net_12/pos/';
%             if ~exist(folder, 'dir')
%                 mkdir(folder)
%             end
%             temp = num2str(num_image);
%             file_name = strcat(temp,'.jpg');
%             imwrite(crop_image,fullfile(folder,file_name));
%             num_image = num_image+1;

            bbox.xmin = 0;
            bbox.ymin = 0;
            bbox.xmax = 0;
            bbox.ymax = 0;
            bbox.xlefteye = 0;
            bbox.ylefteye = 0;
            bbox.xrighteye = 0;
            bbox.yrighteye = 0;
            bbox.xnose = 0;
            bbox.ynose = 0;
            bbox.xleftmouth = 0;
            bbox.yleftmouth = 0;
            bbox.xrightmouth = 0;
            bbox.yrightmouth = 0;

            if ~line(5)==0
                bbox.xmin = line(6);
                bbox.ymin = line(7);
                bbox.xmax = line(8);
                bbox.ymax = line(9);
            end

            data_example{count,1} = crop_image;
            data_example{count,2} = num2str(line(5));
            data_example{count,3} = struct2array(bbox)';
            count = count+1;
            %I_many(:,:,:,1) = data_example.image{1};
        end

    end



end