function [X,T1,T2] = preprocessMiniBatch(dataX,dataT1,dataT2)
    
    % Extract image data from cell and concatenate
    X = cat(4,dataX{:});
    % Extract label data from cell and concatenate
    T1 = cat(2,dataT1{:});
    % Extract angle data from cell and concatenate
    T2 = cat(2,dataT2{:});
        
    % One-hot encode labels
    T1 = onehotencode(T1,1);
        
end