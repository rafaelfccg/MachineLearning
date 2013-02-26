
clear all

max_img_num = 5000;
min_img_num = 10;
max_cls = 100;
min_cls = 5;

%%% train dictionary and get codes  
%

imgdir = 'F:\Datasets\MobileProductSearch\Whole_Dataset\AmazonProductSet\DemoDataSet_resized\';
resdir = 'F:\Results\MobileVisualSearch\gist\';

% category list
cate_list_file = 'F:\Datasets\MobileProductSearch\Whole_Dataset\AmazonProductSet\test_categories.txt';
fid = fopen(cate_list_file);
cate_list = textscan(fid, '%s');
fclose(fid);
cate_list = cate_list{1,1};

for id=1:size(cate_list,1)
    
    gist_feat_file = [resdir cate_list{id,1} '.mat'];
    if(exist(gist_feat_file,'file') == 0)
        continue;
    end
    
    % load training samples
    gists = load(gist_feat_file);
    X = gists.gist'; % put sample in columns
    
    % load image name
    curdir = [imgdir cate_list{id,1} '\'];
    imgnames = dir([curdir '*.jpg']);
    img_num = min(size(imgnames,1), max_img_num);
    if(size(imgnames,1) ~= size(X,2))
        disp('feature and image number not same');
        pause
        continue;
    end
    
    % create results dir
    res_img_dir = [resdir cate_list{i,1} '\'];
    mkdir(res_img_dir);
    
    fprintf('Start to learn dictionary.\n');
    dict_size = ((max_cls-min_cls)*img_num + max_img_num*min_cls-min_img_num*max_cls) / (max_img_num-min_img_num);
    tic
    [C,A] = vl_kmeans(gists, dict_size, 'verbose', 'distance', 'l2', 'algorithm', 'elkan');
    t=toc;
    fprintf('time of computation for Dictionary Learning: %f\n',t);

    % create dir for each dictionary item
    group_dirs = cell(dict_size, 1);
    for i=1:dict_size
        res_dir = [res_img_dir num2str(i) '\'];
        mkdir(res_dir);
        group_dirs{i,1} = res_dir;
    end

    % split images into groups based on distance to dictionary items
    for i=1:size(X,2)
        % copy it to dir
        copyfile([imgdir imgnames(i,1).name], group_dirs{A(1,i),1});
    end
    
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