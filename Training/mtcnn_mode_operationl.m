function [Y1,Y2,state]=mtcnn_mode_operationl(parameters,X,doTraining,state)

% conv_1
    weights = parameters.features_conv1_weight;
    bias = parameters.features_conv1_bias;
    Y = dlconv(X,weights,bias);
    
    % Batch normalization, ReLU - batchnorm1, relu1
    offset = parameters.batchnorm1.Offset;
    scale = parameters.batchnorm1.Scale;
    trainedMean = state.batchnorm1.TrainedMean;
    trainedVariance = state.batchnorm1.TrainedVariance;
    
    if doTraining
        % Normalize data across all observations for each channel independently
        % To speed up training of the convolutional neural network and reduce the sensitivity to network initialization, 
        % use batch normalization between convolution and nonlinear operations such as relu.
        [Y,trainedMean,trainedVariance] = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
        
        % Update state
        state.batchnorm1.TrainedMean = trainedMean;
        state.batchnorm1.TrainedVariance = trainedVariance;
    else
        Y = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
    end
        Y = prelu(Y,parameters.features_prelu1.weight);


% conv_2
    weights = parameters.features_conv2_weight;
    bias = parameters.features_conv2_bias;
    Y = dlconv(Y,weights,bias);
    
    % Batch normalization, ReLU - batchnorm2, relu2
    offset = parameters.batchnorm2.Offset;
    scale = parameters.batchnorm2.Scale;
    trainedMean = state.batchnorm2.TrainedMean;
    trainedVariance = state.batchnorm2.TrainedVariance;
    
    if doTraining
        [Y,trainedMean,trainedVariance] = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
        
        % Update state
        state.batchnorm2.TrainedMean = trainedMean;
        state.batchnorm2.TrainedVariance = trainedVariance;
    else
        Y = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
    end
    
        Y = prelu(Y,parameters.features_prelu2.weight);


% conv_3
    weights = parameters.features_conv3_weight;
    bias = parameters.features_conv3_bias;
    Y = dlconv(Y,weights,bias);
    
    % Batch normalization, ReLU - batchnorm2, relu2
    offset = parameters.batchnorm3.Offset;
    scale = parameters.batchnorm3.Scale;
    trainedMean = state.batchnorm3.TrainedMean;
    trainedVariance = state.batchnorm3.TrainedVariance;
    
    if doTraining
        [Y,trainedMean,trainedVariance] = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
        
        % Update state
        state.batchnorm3.TrainedMean = trainedMean;
        state.batchnorm3.TrainedVariance = trainedVariance;
    else
        Y = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
    end
    
        Y = prelu(Y,parameters.features_prelu3.weight);


% conv_4_1
    weights = parameters.conv4_1_weight;
    bias = parameters.conv4_1_bias;
    Y1 = dlconv(Y,weights,bias);
    
    % Batch normalization, ReLU - batchnorm2, relu2
    offset = parameters.batchnorm4_1.Offset;
    scale = parameters.batchnorm4_1.Scale;
    trainedMean = state.batchnorm4_1.TrainedMean;
    trainedVariance = state.batchnorm4_1.TrainedVariance;
    
    if doTraining
        [Y1,trainedMean,trainedVariance] = batchnorm(Y1,offset,scale,trainedMean,trainedVariance);
        
        % Update state
        state.batchnorm3.TrainedMean = trainedMean;
        state.batchnorm3.TrainedVariance = trainedVariance;
    else
        Y1 = batchnorm(Y1,offset,scale,trainedMean,trainedVariance);
    end
    
    Y1 = softmax(Y1);

% conv_4_2
    weights = parameters.conv4_2_weight;
    bias = parameters.conv4_2_bias;
    Y2 = dlconv(Y,weights,bias);
    
    % Batch normalization, ReLU - batchnorm2, relu2
    offset = parameters.batchnorm4_2.Offset;
    scale = parameters.batchnorm4_2.Scale;
    trainedMean = state.batchnorm4_2.TrainedMean;
    trainedVariance = state.batchnorm4_2.TrainedVariance;
    
    if doTraining
        [Y2,trainedMean,trainedVariance] = batchnorm(Y2,offset,scale,trainedMean,trainedVariance);
        
        % Update state
        state.batchnorm3.TrainedMean = trainedMean;
        state.batchnorm3.TrainedVariance = trainedVariance;
    else
        Y2 = batchnorm(Y2,offset,scale,trainedMean,trainedVariance);
    end
    


end