

% train a multivariate linear regressor

labelfile = 'E:\Datasets\Kaggle_Galaxy\training_solutions_rev1.csv';

% form training data
labels = parseCSV(labelfile);

% load training features
gistdir = 'E:\Datasets\Kaggle_Galaxy\features\kaggle-trainfeatures-denoised\images_training_rev1_features\gist\';

len = 100;
gists = zeros(len, 512);
for i=1:len
    
    curfile = [gistdir num2str(labels(i,1)) '.feat'];
    gists(i, :) = load(curfile)';
    
    fprintf('%d/%d\n', i, len);
    
end

% set up model
gists = [ones(len,1) gists];
% train model
[beta, sigma, resid] = mvregress(gists, labels(1:len, 2:end));

% test model



