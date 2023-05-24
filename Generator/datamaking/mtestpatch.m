% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/08/13
% Southwest university

% making maskrcnn test data
% folder strcture
%>>>>>>|->test
%         |-patch0
%             |-images
%                 |-*.png

clc,clear,close all;
dbstop if error
%% load the data and enter the folder of test
datadir = '/home/hadmatter/桌面/homogeneous_nanoparitcle_cluster/Datasets/';
cd(datadir);
load('data.mat');
testdir = [datadir,'mrcnn/dataset/test'];
cd(testdir)
dirOutput=dir();
foldernum = length(dirOutput)-2;

%%
for i = 1:200
    imtemp = img(i,:);
    imtemp = reshape(imtemp,890,1280);
    X = cut6patch(imtemp);
    [patch1,patch2,patch3,patch4,patch5,patch6] = deal(X{:});
    for jj = 1:6
        subfolder = ['patch',num2str(foldernum)];
        mkdir(subfolder);
        mkdir([subfolder,'/images/']);
        imagename = [subfolder,'/images/',subfolder,'.png'];
        imt = eval(['patch',num2str(jj)]);
        imt = imresize(imt,[256,256],'nearest');
        imwrite(imt,imagename);
        foldernum = foldernum+1;
        foldernum
    end
end

%%

function X = cut6patch(img)
basicsize = 890/2;
cutx = [1,418,836];
cuty = [1,446];
cutp1 = [cutx(1) cuty(1) basicsize basicsize];
cutp2 = [cutx(2) cuty(1) basicsize basicsize];
cutp3 = [cutx(3) cuty(1) basicsize basicsize];
cutp4 = [cutx(1) cuty(2) basicsize basicsize];
cutp5 = [cutx(2) cuty(2) basicsize basicsize];
cutp6 = [cutx(3) cuty(2) basicsize basicsize];

X = {imcrop(img,cutp1),imcrop(img,cutp2),imcrop(img,cutp3)...
    ,imcrop(img,cutp4),imcrop(img,cutp5),imcrop(img,cutp6)};
end