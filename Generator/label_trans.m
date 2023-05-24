% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/09/24
% Southwest university


clc,clear,close all
% label convention
workdir = '/home/hadmatter/桌面/homogeneous_nanoparitcle_cluster/generator';
load unet.mat

ulb = result(:,2);

im1 = ulb{1};
%%

%for i = 1:200
i = 1;
    imt = ulb{i};
    most = mode(imt(:));
    imt2 = imt;
    imt2(find(imt==most))=0;
    
    
    
    
%end
