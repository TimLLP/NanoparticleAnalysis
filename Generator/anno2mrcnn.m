%% annot2mrcnn
% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/11/4
% Southwest university 

close all,clc

clear;
global annodir annodir1
annodir = '/home/hadmatter/Desktop/labeler_results/labels/Unet2';
annodir1 = '/home/hadmatter/Desktop/set3';

outputdir = '/home/hadmatter/Desktop/homogeneous_nanoparitcle_cluster/finaltest/';
bmap=zeros(256,256);

%
% if the masks folders are in absence, run the following code
% for i = 0:1199
%     subfolder = ['patch',num2str(i)];
%     cd(subfolder)
%     mkdir('masks');
%     cd(outputdir)
% end

          

%% GIMP labeled iamge

cd(annodir1);
img_path = dir();
img_path([1,2])=[];
for i = 1:length(img_path)
    cd(annodir1);
    img_name = img_path(i).name;
    num = str2num(img_name(1:end-4));
    
    iml = imread(img_path(i).name);% read img
    if ndims(iml) == 3
        iml = im2double(rgb2gray(iml));
    end
    
    iml(find(iml>0.8))=1;
    iml = uint8(iml);
    
    cd(outputdir)
    startf_n = (num-1)*6;
    X = cut6patch(iml);
    [patch1,patch2,patch3,patch4,patch5,patch6] = deal(X{:});
    for jj = 1:6
        subfolder = ['patch',num2str(startf_n)];
        imt = eval(['patch',num2str(jj)]);
        imt = imresize(imt,[256,256],'nearest');
        CC = bwconncomp(imt,4);
        list = CC.PixelIdxList;
        li_l=length(list);
        for kk = 1:li_l
            if length(list{kk})>100
                gtname = [subfolder,'/masks/',subfolder,'_',num2str(jj),'_',num2str(kk),'.png'];
                bmapt = bmap;
                bmapt(list{kk})=1;
                imwrite(bmapt,gtname);
            end
        end
        startf_n
        startf_n = startf_n+1;
    end
end


%%

cd("/home/hadmatter/PycharmProjects/Mask_RCNN/samples/final/val");
flist = dir();
flist([1,2])=[];
for ff=1:length(flist)
    subfileinfo = dir(fullfile("/home/hadmatter/PycharmProjects/Mask_RCNN/samples/final/val",flist(ff).name,'masks'));%获取子文件夹的信息
    if length(subfileinfo) == 2%判断是否为空，因为matlab有.和..，所以空文件夹的信息长度为2
        nn = flist(ff).name;
        disp(nn(6:end));
        rmdir(nn,'s');
    end
end



%% MATLAB laebled image


i = 22
delallmasks(i)
subdir = 'PixelLabelData';
cd(annodir)
cd(['img',num2str(i,'%03d'),'/',subdir]);
img_path_list = dir('*.png');
iml = imread(img_path_list.name);% read img
iml(find(iml~=0))=1;
iml = uint8(iml);

%imshow(iml,[]);
cd(outputdir)
startf_n = (i-1)*6;
X = cut6patch(iml);
[patch1,patch2,patch3,patch4,patch5,patch6] = deal(X{:});
for jj = 1:6
    subfolder = ['patch',num2str(startf_n)];
    imt = eval(['patch',num2str(jj)]);
    imt = imresize(imt,[256,256],'nearest');
    CC = bwconncomp(imt,4);
    list = CC.PixelIdxList;
    li_l=length(list);
    for kk = 1:li_l
        if length(list{kk})>100
            gtname = [subfolder,'/masks/',subfolder,'_',num2str(jj),'_',num2str(kk),'.png'];
            bmapt = bmap;
            bmapt(list{kk})=1;
            imwrite(bmapt,gtname);
        end
    end
    startf_n
    startf_n = startf_n+1;
end

%%

function delallmasks(num)

outputdir = '/home/hadmatter/Desktop/homogeneous_nanoparitcle_cluster/finaltest/';
cd(outputdir)
anchor = (num-1)*6;
for pp = 1:6
    subfolder = ['patch',num2str(anchor)];
    cd (subfolder);
    rmdir masks s
    mkdir masks
    cd ..
    anchor  = anchor +1;
end
end


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