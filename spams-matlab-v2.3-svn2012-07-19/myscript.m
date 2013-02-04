
clear all

%%% train dictionary and get codes  
%
% load training samples
gists = load('H:\MobileVisualSearch\gist\cookware.mat');
X = gists.gist'; % put sample in columns

% load image name
imgdir = 'H:\AmazonProductData\DemoDataSet_resized\cookwar\';
imgnames = load('H:\MobileVisualSearch\gist\cookware_filenames.mat');
imgnames = imgnames.readlist;

param.K=50;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.numThreads=4; % number of threads
param.batchsize=400;

param.iter=1000;  % let us see what happens after 1000 iterations.

fprintf('Start to learn dictionary.\n');
tic
D = mexTrainDL(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);

% create dictionary for each dictionary item
group_dirs = cell(param.K,1);
for i=1:param.K
    res_dir = ['H:\MobileVisualSearch\gist\cookware\' num2str(i) '\'];
    mkdir(res_dir);
    group_dirs{i,1} = res_dir;
end

% split images into groups based on distance to dictionary items
for i=1:size(X,2)
    diff = repmat(X(:,i), 1, param.K) - D;
    diff = sqrt(sum(diff.^2,1));
    [min_dist, min_id] = min(diff);
    % copy it to dir
    copyfile([imgdir imgnames(i,1).name], group_dirs{min_id,1});
end


param.approx=0;
fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
% ImD=displayPatches(D);
% subplot(1,3,1);
% imagesc(ImD); colormap('gray');
fprintf('objective function: %f\n',R);
drawnow;