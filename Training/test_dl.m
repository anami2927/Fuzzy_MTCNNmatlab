layer2 = [
    imageInputLayer([9 36 1],'Normalization','none','Name','input1-fcc')
    convolution2dLayer([7,7],64,'Name','conv1-fcc')
    batchNormalizationLayer('Name','bn1-fcc')
    reluLayer('Name','relu1-fcc')
    maxPooling2dLayer([2 2],'Name','pool')
    fullyConnectedLayer(1,'Name','fc1')];
lgraph = layerGraph(layer2);
dlnet = dlnetwork(lgraph);
% Input
a = rand(9,36,1,10);
a = dlarray(a,'SSCB');
[loss,gradients] = dlfeval(@compute_gradient,dlnet,a);
function [loss,gradients]=compute_gradient(dlnet,a)
    a_pre = forward(dlnet,a);
    % output
    b = rand(1,10);
    loss = mse(a_pre,b);
    gradients = dlgradient(dlarray(loss),dlnet.Learnables);%automatic gradient
end