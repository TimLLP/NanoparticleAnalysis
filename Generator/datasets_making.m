% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/08/13
% Southwest university

clc,clear,close all;
dbstop if error
global datadir unetimgdir unetlabdir
datadir = 'F:\mproject\datasets\generator';
unetimgdir = [datadir,'F:\mproject\datasets\generator'];
unetlabdir = [datadir,'F:\mproject\±ê×¢\ÒÑ±ê×¢\label'];
cd(datadir)

% generate two types of datasets
%option = 'unet'
option = 'mrcnn'

for ii =1:18
    imnum = ii;
    cd(['gra',num2str(imnum,'%03d')]);
    currentFolder = pwd;
    file=dir([currentFolder,'/*.mat']);% the mat file dir and name
    filename = [currentFolder,'/', num2str(imnum,'%03d'),'.mat'];
    if length(file)>0
        load(file.name);% obtain the mat file
    end
    %load(filename)
    mkdir img
    mkdir mask
    % % show all synthesized map
    % figure;
    % for  ii=1:20
    %     subplot(4,5,ii)
    %     imshow(parameter(ii).map)
    % end
    switch option
        case 'unet'
            unetpatch(parameter)
        case 'mrcnn'
            mskrcnnpatch(parameter)
    end
    cd(datadir)
end

%% function file
function mskrcnnpatch(parameter)
% making the image and mask file
for k = 1:20
    curpath = cd;
    mrimg = parameter(k).map;
    mrmask = parameter(k).map_mask;
    
    mrX = cut6patch(mrimg);
    [mrpatch1,mrpatch2,mrpatch3,mrpatch4,mrpatch5,mrpatch6] = deal(mrX{:});
    mrX_m = cut6patch(mrmask);
    [mrp1_m,mrp2_m,mrp3_m,mrp4_m,mrp5_m,mrp6_m] = deal(mrX_m{:});
    
    for j = 1:3
        imname = [curpath(end-5:end),'_',num2str(k,'%02d'),'_',num2str(j),'.png']
        % save the img
        cd([curpath,'/img'])
        imwrite(eval(['mrpatch',num2str(j)]),imname);
        
        % save the mask
        % remove highly occluded particle
        mrpatchtemp = eval(['mrp',num2str(j),'_m']);
        mrpatchtable = tabulate(mrpatchtemp(:));
        mrremove = mrpatchtable(find(mrpatchtable(:,3)<1),1);
        mrpatchtemp(find(ismember(mrpatchtemp,mrremove)==1))=0;
        % revaule the mask
        
        cd([curpath,'/mask'])
        mkdir(imname(1:end-4));cd(imname(1:end-4));
        ptcnum = unique(mrpatchtemp(:));
        for i = 1:length(ptcnum)
            if i~=1
                imzero = double(zeros(size(mrpatchtemp)));
                imzero(find(mrpatchtemp==ptcnum(i)))=1;
                mskname = [imname(1:end-4),'_m',num2str(i,'%03d'),'.png'];
                imwrite(imzero,mskname);
            end
        end
    end
    cd(curpath)
end   
end


function unetpatch(parameter)
global unetimgdir unetlabdir
% output the number of the .png files in unet/image
cd(unetimgdir)
dirOutput=dir('*.png');
fileNames={dirOutput.name}';
filenum = numel(fileNames);
option = repmat([1;2;3;1;2],4,1);

for kk = 1:20
    img = parameter(kk).map;
    mask = parameter(kk).map_mask;
    %imshowpair(img,unetmask);
    
    X = cut6patch(img);
    [patch1,patch2,patch3,patch4,patch5,patch6] = deal(X{:});
    X_m = cut6patch(mask);
    [patch1_m,patch2_m,patch3_m,patch4_m,patch5_m,patch6_m] = deal(X_m{:});

    for jj = 1:6
        if filenum==0
            imagename = ['0.png'];
        else
            imagename = [num2str(filenum),'.png']
        end
        % remove highly occluded particle
        patchtemp = eval(['patch',num2str(jj),'_m']);
        patchtable = tabulate(patchtemp(:));
        remove = patchtable(find(patchtable(:,3)<1),1);
        patchtemp(find(ismember(patchtemp,remove)==1))=0;
        % 
        [maskx,masky] = gradient(double(patchtemp));
        unetmask = zeros(size(patchtemp));
        unetmask(find(maskx~=0))=1;
        unetmask(find(masky~=0))=1;
        
        % saved the image
        cd(unetimgdir)
        imwrite(eval(['patch',num2str(jj)]),imagename);
        cd ..
        cd(unetlabdir)
        imwrite(unetmask,imagename);    
        filenum = filenum+1;
    end
end
end


function X = cut6patch(img)
basicsize = 890/2-1;
testim = zeros(890,1280);
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
