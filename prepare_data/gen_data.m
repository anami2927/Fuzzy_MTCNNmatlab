
function out = gen_data(input_size)
    data_dir = 'data';
    sizenet = input_size;

    if sizenet == 12
        net ='p_net';
    elseif sizenet ==24 
        net ='r_net';
    else
        net = 'o_net';
    end

    output_dir = strcat(data_dir,"/",net);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    if sizenet == 12
       read_files(1) =  strcat(output_dir,"/train_p_net_pos.txt");
       read_files(2) =  strcat(output_dir,"/train_p_net_part.txt");
       read_files(3) =  strcat(output_dir,"/train_p_net_neg.txt");
       read_files(4) =  strcat(output_dir,"/train_p_net_landmark.txt");

      % Generate Pnet data
       get_p_net_data(data_dir,12,read_files(1:3));

      % Generate landmark file
       gen_landmark
       
    end

end