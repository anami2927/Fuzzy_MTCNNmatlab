net = googlenet;
% Convert the network to a layer graph and remove the layers used for classification using removeLayers.
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,["prob" "output"]);

% Convert the network to a dlnetwork object.
dlnet = dlnetwork(lgraph)




