% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/09/18
% Southwest university

% making training sample of unet

clear,close all,clc;
loaddir = '/home/hadmatter/桌面/homogeneous_nanoparitcle_cluster/generator/dataset_making/';
imgdir = '/media/hadmatter/Adamberg/备份/homogeneous_nanoparitcle_cluster/Datasets/';
unetcd = '/home/hadmatter/桌面/homogeneous_nanoparitcle_cluster/Datasets/Unet/'
global uimdir ulbdir
uimdir = '/home/hadmatter/桌面/homogeneous_nanoparitcle_cluster/Datasets/Unet/image/'
ulbdir = '/home/hadmatter/桌面/homogeneous_nanoparitcle_cluster/Datasets/Unet/label/'

% load two variables: canlabel_cos; canlabel_l2
load([loaddir,'data0.mat']);
% sort and filter
[xx,yy] = conversion_s(canlabel_l2, canlabel_cos, 4);

% cut and save
for i = 1:length(xx)
    filedir = [imgdir,'img',num2str(xx(i),'%03d'),'/'];
    li = dir([filedir,'*.mat']);
    ivar = matfile([li.folder,'/',li.name]);
    parameter = ivar.parameter;
    para1 = parameter(yy(i));
    unetpatch(parameter)
    disp(num2str(i));
end

%% Function part

function [xx,yy] = conversion_r(canlabel_l2, canlabel_cos)
%%%%%%%%%%%%%%%%%%%%
% output the selected images from the result of clustering
%%%%%%%%%%%%%%%%%
% format conversion
canlabel_l2 = single(canlabel_l2);
canlabel_cos = single(canlabel_cos);
% find non-repetitive elements and their frequency
lb_l2 = unique(canlabel_l2);% l2 cluster
[m1,n1] = hist(canlabel_l2(:),lb_l2);
lb_cos = unique(canlabel_cos);% cos cluster
[m2,n2] = hist(canlabel_cos(:),lb_cos);
% sort according to their frequency
B1 = double(sortrows([m1',n1], 1, 'descend'));
B2 = double(sortrows([m2',n2], 1, 'descend'));
% normalization
B1(:,1) = B1(:,1)/sum(B1(:,1));
B2(:,1) = B2(:,1)/sum(B2(:,1));

% Character split for finding the cooresponding imgs
z1 = num2str(B1(:,2)); z1 = str2num(z1(:,end));
y1 = num2str(B1(:,2)); y1 = str2num(y1(:,end-2:end-1));
x1 = num2str(B1(:,2)); x1 = str2num(x1(:,1:end-3));
C1 = [x1,y1,z1,B1(:,1)];

z2 = num2str(B2(:,2)); z2 = str2num(z2(:,end));
y2 = num2str(B2(:,2)); y2 = str2num(y2(:,end-2:end-1));
x2 = num2str(B2(:,2)); x2 = str2num(x2(:,1:end-3));
C2 = [x2,y2,z2,B2(:,1)];

% combine these two distance
weight1 = zeros(200,20);
weight2 = zeros(200,20);
for i = 1:length(C1)
    x = C1(i,1);
    y = C1(i,2);
    weight1(x,y) = weight1(x,y) + C1(i,4);
end
for i = 1:length(C2)
    x = C2(i,1);
    y = C2(i,2);
    weight2(x,y) = weight2(x,y) + C2(i,4);
end

weight = weight1*0.5 + weight2*0.5;
number = sum(sum(weight~=0));
% normalization
weight = weight./sum(weight(:));
imshow(weight,[]);
colormap('jet');
[~,I]= sort(reshape(weight,1,numel(weight)),'descend');
B(I)=1:numel(weight);
weightind = reshape(B,size(weight));
xx = []; yy = [];
for i = 1:number
    [xt,yt] = find(weightind == i);
    xx = [xx;xt];
    yy = [yy;yt];
end
end

function [xx,yy] = conversion_s(canlabel_l2, canlabel_cos, topk)
% format conversion
canlabel_l2 = single(canlabel_l2);
canlabel_cos = single(canlabel_cos);
xx = [];yy = [];

for i = 1:200
    mat = [canlabel_l2(2*i-1,:),canlabel_l2(2*i,:),...
        canlabel_cos(2*i-1,:),canlabel_cos(2*i,:)];
    mat_lb = unique(mat);
    [m,n] = hist(mat(:),mat_lb);
    B = double(sortrows([m',n'], 1, 'descend'));
    B(:,1) = B(:,1)/sum(B(:,1));
    lbstr = num2str(B(:,2));
    C =[str2num(lbstr(:,1:end-3)),str2num(lbstr(:,end-2:end-1)),...
        str2num(lbstr(:,end)),B(:,1)];
    
    weight = zeros(1,20);
    for j = 1:length(C)
        y = C(j,2);
        weight(y) = weight(y) + C(j,4);
    end
    number = sum(weight~=0);
    
    weight = weight./sum(weight(:));
    [~,I]= sort(weight,'descend');
    
    if number >= topk
        y = I(1:topk);
        x = ones(size(y))*i;
        yy = [yy;y'];
        xx = [xx;x'];  
    else
        disp(['topk should be lower than',num2str(number)]);
    end
    
end    
end


%%%
function unetpatch(parameter)

global uimdir ulbdir
img = parameter.map;
mask = parameter.map_mask;
%imshowpair(img,unetmask);

cd(uimdir)
dirOutput=dir('*.png');
fileNames={dirOutput.name}';
filenum = numel(fileNames);

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
    remove = patchtable(find(patchtable(:,3)<0.01),1);
    patchtemp(find(ismember(patchtemp,remove)==1))=0;
    % 
    [maskx,masky] = gradient(double(patchtemp));
    unetmask = zeros(size(patchtemp));
    unetmask(find(maskx~=0))=1;
    unetmask(find(masky~=0))=1;
    
    % saved the image
    cd(uimdir)
    imt = eval(['patch',num2str(jj)]);
    imt = imresize(imt,[256,256],'nearest');
    imwrite(imt,imagename);
    cd ..
    cd(ulbdir)
    unetmask = imresize(unetmask,[256,256],'nearest');
    imwrite(unetmask,imagename);
    filenum = filenum+1;
end
end

function X = cut6patch(img)
%%%%%%%%%%%
%cut picture into 6 patches
%%%%%%%%%%
basicsize = 890/2-1;
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



