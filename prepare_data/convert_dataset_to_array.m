function [XTrain, T2Train] = convert_dataset_to_array(dataset)
    N = numel(dataset(:,1));
    for i=1:N
        XTrain(:,:,:,i) = dataset{i,1};
        T2Train(i,:) = dataset{i,3}';
    end
        XTrain = single(XTrain);
end