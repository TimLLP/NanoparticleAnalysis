% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/06/13
% Southwest university

close all,clc
clear;
unetdir = '/home/hadmatter/Desktop/labeler_results/labels/Unet2';
cd(unetdir)

% %% If all .png files are in one folder, change them to separate files
% % only use one time
% for i = 1:200
%     mkdir(['img',num2str(i,'%03d')]);
%     fname = ['img',num2str(i,'%03d')];
%     imgname = [num2str(i,'%03d'),'.png'];
%     movefile(imgname,fname);
% end

%%

id = 25;

cd(unetdir);
fname = ['img',num2str(id,'%03d')];
cd(fname)
imageLabeler(cd)

%%






