% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/09/05
% Southwest university

dbstop if error
close all,clc,clear;
loadcd = 'F:\mproject\datasets\ori';
savecd = 'F:\mproject\datasets\img';
global imgset
cd(loadcd)
list = dir(loadcd);
%%
for ii = 6:length(list)
    if (list(ii).isdir&&startsWith(list(ii).name,'img'))
        pref = [loadcd,'/',list(ii).name,'/'];
        li = dir([pref,'*.mat']);
        ivar = matfile([li.folder,'/',li.name]);
        parameter = ivar.parameter;
        imgset = {parameter(:).map};
       % printf(imgset);
        clearvars parameter
        disp(['载入第',list(ii).name,'文件']);
    end
    for jj = 1:length(imgset)
        img = imgset{jj};
        imgname = [li.name(6:end-4),'_',num2str(jj,'%02d'),'.png'];
        cd(savecd);
        imwrite(img,imgname);
    end
    disp(['写入第',list(ii).name,'文件']);
end

%%
cd(savecd);
allimg = dir('*.png');
length(allimg)
